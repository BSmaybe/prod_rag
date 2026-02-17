from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.db import db_conn, ensure_schema, enqueue_outbox_event, upsert_ticket_event
from api.schemas.servicedesk import TicketsBatchIn
from core.vectordb import upsert_tickets
from etl.anonymize import anonymize_text

router = APIRouter(prefix="/sd", tags=["service-desk"])
log = logging.getLogger("rag.sd")


class TicketsBatchOut(BaseModel):
    accepted: int
    stored_file: str
    reindex_started: bool
    reindex_result: Optional[dict] = None


KB_COLLECTION = os.getenv("COLLECTION_NAME", "kb_tickets")


# =========================
# Helpers
# =========================

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def _normalize_text(s: str) -> str:
    """
    Единая нормализация текста:
    - html entities (&nbsp; и т.п.) -> юникод
    - \xa0 -> пробел
    - схлопывание пробелов
    """
    if not s:
        return ""
    s = html.unescape(s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_rtf(text: str) -> str:
    """
    Упрощённая очистка:
    - html теги
    - rtf-эскейпы вида \par \b0 \u1234 ...
    - декодирование HTML entities (&nbsp; и т.п.)
    """
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"<[^>]+>", " ", t)               # html tags
    t = re.sub(r"\\[a-zA-Z]+\d*", " ", t)        # rtf escapes
    t = re.sub(r"[\{\}]", " ", t)                # braces
    return _normalize_text(t)


def _get_in(d: Dict[str, Any], path: List[str], default: Any = "") -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _dump_raw(raw_dir: Path, raw_bytes: bytes, content_type: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    p = raw_dir / f"sd_raw_{ts}.bin"
    meta = raw_dir / f"sd_raw_{ts}.meta.txt"
    p.write_bytes(raw_bytes)
    meta.write_text(
        f"content-type: {content_type}\nlen: {len(raw_bytes)}\n",
        encoding="utf-8",
    )
    return p


def _try_parse_json_from_anything(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Naumen шлёт JSON так:
    {
      "UUID": "...",
      "type": "...",
      "message": {
          "text": "{ \"header\": {...}, \"serviceCall\": {...} }"
      }
    }
    """
    s = raw_bytes.decode("utf-8", errors="replace").strip()

    try:
        outer = json.loads(s)
    except Exception:
        outer = None

    if isinstance(outer, dict):
        # 1) главное — message.text
        try:
            msg = outer.get("message")
            if isinstance(msg, dict):
                txt = msg.get("text")
                if isinstance(txt, str):
                    t = txt.strip()
                    if t.startswith("{") and t.endswith("}"):
                        inner = json.loads(t)
                        if isinstance(inner, dict):
                            return inner
        except Exception:
            pass

        # 2) вдруг text на верхнем уровне
        try:
            txt = outer.get("text")
            if isinstance(txt, str):
                t = txt.strip()
                if t.startswith("{") and t.endswith("}"):
                    inner = json.loads(t)
                    if isinstance(inner, dict):
                        return inner
        except Exception:
            pass

        # 3) batch {"tickets":[...]}
        return outer

    # 4) fallback — JSON внутри строки
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        chunk = s[i:j + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def _has_servicecall_uuid(body: Dict[str, Any]) -> bool:
    if not isinstance(body, dict):
        return False
    sc = body.get("serviceCall") or {}
    hdr_sc = _get_in(body, ["header", "serviceCall"], default={})

    sc_uuid = ""
    if isinstance(sc, dict):
        sc_uuid = _safe_str(sc.get("UUID") or "")
    if not sc_uuid and isinstance(hdr_sc, dict):
        sc_uuid = _safe_str(hdr_sc.get("UUID") or "")
    return bool(sc_uuid.strip())


def _extract_event(body: Dict[str, Any]) -> Tuple[str, str, Dict[str, str]]:
    """
    Собираем плоский event из serviceCall.
    Возвращаем (issue_key, description_plain, flat)
    """
    hdr_sc = _get_in(body, ["header", "serviceCall"], default={})
    sc = body.get("serviceCall") or {}

    sc_uuid = _safe_str(sc.get("UUID") or (hdr_sc.get("UUID") if isinstance(hdr_sc, dict) else "") or "")
    issue_key = _safe_str((hdr_sc.get("title") if isinstance(hdr_sc, dict) else None) or sc.get("title") or sc_uuid or "unknown")
    state = _safe_str(sc.get("state") or (hdr_sc.get("state") if isinstance(hdr_sc, dict) else "") or "")

    desc_rtf = (
        _safe_str(sc.get("descriptionInRTF"))
        or _safe_str(sc.get("techDesc"))
        or _safe_str(sc.get("description"))
        or _safe_str(body.get("descriptionInRTF"))
        or _safe_str(body.get("description"))
    )
    text = _strip_rtf(desc_rtf)

    route = _safe_str(_get_in(sc, ["route", "title"]))
    slm_service = _safe_str(_get_in(sc, ["slmService", "title"]))
    priority = _safe_str(_get_in(sc, ["customPriority", "title"]))
    team = _safe_str(_get_in(sc, ["responsibleTeam", "title"]))

    client_emp = _safe_str(_get_in(sc, ["clientEmployee", "title"]))
    client_phone = _safe_str(_get_in(sc, ["clientEmployee", "mobilePhoneNumber"]))
    client_email = _safe_str(_get_in(sc, ["clientEmployee", "email"]))

    reg_dt = _safe_str(sc.get("registrationDate") or "")
    creation_dt = _safe_str(sc.get("creationDate") or "")

    # totalValue -> tv_*
    tv_list = sc.get("totalValue") or []
    tv_flat: Dict[str, str] = {}
    if isinstance(tv_list, list):
        for item in tv_list:
            if not isinstance(item, dict):
                continue
            k = _safe_str(item.get("title")).strip()
            if not k:
                continue
            v = item.get("textValue")
            if v is None or _safe_str(v).strip() == "":
                v = item.get("value")
            tv_flat[k] = _strip_rtf(_safe_str(v))

    flat: Dict[str, str] = {
        "sc_uuid": sc_uuid,
        "issue_key": issue_key,
        "state": state,
        "route": route,
        "slm_service": slm_service,
        "priority": priority,
        "responsible_team": team,
        "client_employee": client_emp,
        "client_phone": client_phone,
        "client_email": client_email,
        "registration_date": reg_dt,
        "creation_date": creation_dt,
        "description_plain": text,
    }
    for k, v in tv_flat.items():
        flat[f"tv_{k}"] = v

    return issue_key, text, flat


def _last_csv_row(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
            if not rows:
                return None
            return rows[-1]
    except Exception:
        return None


def _append_events_csv(path: Path, flat: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "ts",
        "issue_key",
        "sc_uuid",
        "state",
        "route",
        "slm_service",
        "priority",
        "responsible_team",
        "client_employee",
        "client_phone",
        "client_email",
        "registration_date",
        "creation_date",
        "description_plain",
    ]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"ts": ts, **flat}

    last = _last_csv_row(path)
    if last is not None:
        cmp_cols = [
            "issue_key", "sc_uuid", "state", "route", "slm_service", "priority",
            "responsible_team", "client_employee", "client_phone", "client_email",
            "registration_date", "creation_date", "description_plain",
        ]
        same = True
        for c in cmp_cols:
            if _safe_str(last.get(c, "")) != _safe_str(row.get(c, "")):
                same = False
                break
        if same:
            return

    if not path.exists():
        tv_cols = sorted([c for c in row.keys() if c.startswith("tv_")])
        cols = base_cols + [c for c in tv_cols if c not in base_cols]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerow({c: row.get(c, "") for c in cols})
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_cols = next(reader, [])

    new_cols = list(existing_cols)
    for k in row.keys():
        if k not in new_cols:
            new_cols.append(k)

    if new_cols != existing_cols:
        with path.open("r", newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=new_cols)
            w.writeheader()
            for r in old_rows:
                w.writerow({c: r.get(c, "") for c in new_cols})
            w.writerow({c: row.get(c, "") for c in new_cols})
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=existing_cols)
        w.writerow({c: row.get(c, "") for c in existing_cols})


def _state_key(flat: Dict[str, str]) -> str:
    sc_uuid = (flat.get("sc_uuid") or "").strip()
    if sc_uuid:
        return sc_uuid
    issue_key = (flat.get("issue_key") or "").strip()
    return issue_key or "unknown"


def _merge_state(prev: Dict[str, Any], flat: Dict[str, str]) -> Dict[str, Any]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = dict(prev or {})
    out.setdefault("updated_at", now)
    out.setdefault("created_at", now)
    out["updated_at"] = now

    for k, v in flat.items():
        v = _safe_str(v).strip()
        if v:
            out[k] = v
        else:
            out.setdefault(k, "")

    ev = {
        "ts": now,
        "state": flat.get("state", ""),
        "route": flat.get("route", ""),
        "priority": flat.get("priority", ""),
        "responsible_team": flat.get("responsible_team", ""),
        "description_plain": (flat.get("description_plain", "") or "")[:500],
    }
    out.setdefault("events", [])
    if isinstance(out["events"], list):
        if not out["events"] or out["events"][-1] != ev:
            out["events"].append(ev)
        out["events"] = out["events"][-50:]
    return out


def _write_state(state_dir: Path, state_key: str, state: Dict[str, Any]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_\-$\.]", "_", state_key) or "unknown"
    p = state_dir / f"{safe}.json"
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _read_state(state_dir: Path, state_key: str) -> Dict[str, Any]:
    safe = re.sub(r"[^a-zA-Z0-9_\-$\.]", "_", state_key) or "unknown"
    p = state_dir / f"{safe}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _same_as_last_event(prev_state: Dict[str, Any], flat: Dict[str, str]) -> bool:
    """
    Дедуп для outbox: если событие по смыслу совпадает с последним — не шлем.
    (ts игнорируем)
    """
    last = None
    if isinstance(prev_state, dict):
        events = prev_state.get("events")
        if isinstance(events, list) and events:
            last = events[-1]
    if not isinstance(last, dict):
        return False

    cur = {
        "state": flat.get("state", ""),
        "route": flat.get("route", ""),
        "priority": flat.get("priority", ""),
        "responsible_team": flat.get("responsible_team", ""),
        "description_plain": (flat.get("description_plain", "") or "")[:500],
    }
    for k, v in cur.items():
        if _safe_str(last.get(k, "")) != _safe_str(v):
            return False
    return True


def _normalize_state(state: str) -> str:
    return (state or "").strip().lower()


def _is_closed_state(state: str) -> bool:
    normalized = _normalize_state(state)
    return normalized in {
        "closed",
        "close",
        "resolved",
        "done",
        "completed",
        "закрыт",
        "закрыта",
        "решен",
        "решён",
    }


def _build_n8n_payload(
    *,
    ticket_id: str,
    text_anonymized: str,
    trace_id: str,
    state: str,
    flat: Dict[str, str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event_type": "ticket_received",
        "ticket_id": ticket_id,
        "issue_key": (flat.get("issue_key") or "").strip(),
        "text": _normalize_text(text_anonymized),  # страховка от &nbsp;/\xa0
        "trace_id": trace_id,
        "state": state,
        "is_closed": _is_closed_state(state),
    }
    if flat.get("route"):
        payload["service"] = flat.get("route")
    if flat.get("slm_service"):
        payload["component"] = flat.get("slm_service")
    return payload


# =========================
# Route
# =========================

@router.post("/tickets", response_model=TicketsBatchOut, status_code=202)
async def ingest_from_sd(request: Request):
    raw = await request.body()
    content_type = request.headers.get("content-type", "")
    request_id = getattr(request.state, "request_id", "-")

    raw_dir = Path(os.getenv("SD_RAW_DIR", "data/sd_raw"))
    events_dir = Path(os.getenv("SD_EVENTS_DIR", "data/sd_events"))
    state_dir = Path(os.getenv("SD_STATE_DIR", "data/sd_state"))

    raw_path = _dump_raw(raw_dir, raw, content_type)
    body = _try_parse_json_from_anything(raw)

    max_batch = _get_env_int("SD_MAX_BATCH", 500)
    max_text_len = _get_env_int("SD_MAX_TEXT_LEN", 10000)
    events_file = events_dir / "sd_events.csv"

    outbox_enqueued = 0
    accepted = 0

    try:
        with db_conn() as conn:
            ensure_schema(conn)

            # =========================
            # Batch: {"tickets":[...]}
            # =========================
            if isinstance(body.get("tickets"), list):
                payload = TicketsBatchIn.model_validate(body)
                if not payload.tickets:
                    raise HTTPException(status_code=400, detail="tickets is empty")
                if len(payload.tickets) > max_batch:
                    raise HTTPException(status_code=413, detail=f"too many tickets (max {max_batch})")

                for t in payload.tickets:
                    ticket_id = (t.issue_key or "").strip()
                    if not ticket_id:
                        continue

                    raw_text = _normalize_text((t.text or "")[:max_text_len])
                    if not raw_text:
                        continue

                    clean_text = anonymize_text(raw_text)

                    flat = {
                        "sc_uuid": "",
                        "issue_key": ticket_id,
                        "state": "new",
                        "route": "",
                        "slm_service": "",
                        "priority": "",
                        "responsible_team": "",
                        "client_employee": "",
                        "client_phone": "",
                        "client_email": "",
                        "registration_date": "",
                        "creation_date": "",
                        "description_plain": clean_text,
                    }

                    key = _state_key(flat)
                    prev = _read_state(state_dir, key)
                    is_dup = _same_as_last_event(prev, flat)

                    _write_state(state_dir, key, _merge_state(prev, flat))
                    _append_events_csv(events_file, flat)

                    upsert_ticket_event(
                        conn,
                        ticket_id=ticket_id,
                        issue_key=ticket_id,
                        status="new",
                        text_anonymized=clean_text,
                        raw_payload={"tickets_item": t.model_dump(), "parsed": body},
                        trace_id=request_id,
                        autocommit=False,
                    )

                    if not is_dup:
                        outbox_id = enqueue_outbox_event(
                            conn,
                            ticket_id=ticket_id,
                            payload=_build_n8n_payload(
                                ticket_id=ticket_id,
                                text_anonymized=clean_text,
                                trace_id=request_id,
                                state="new",
                                flat=flat,
                            ),
                            autocommit=False,
                        )
                        if outbox_id:
                            outbox_enqueued += 1
                            log.info(
                                "outbox_enqueued request_id=%s ticket_id=%s outbox_id=%s",
                                request_id, ticket_id, outbox_id,
                            )
                        else:
                            log.info("outbox_dedup_skip request_id=%s ticket_id=%s", request_id, ticket_id)

                    conn.commit()
                    accepted += 1

            # =========================
            # Single event: {"header":...,"serviceCall":...}
            # =========================
            else:
                if not _has_servicecall_uuid(body):
                    fallback_id = _safe_str(body.get("UUID") or body.get("title") or request_id).strip() or request_id
                    fallback_text = anonymize_text(_normalize_text(json.dumps(body, ensure_ascii=False))[:max_text_len])
                    upsert_ticket_event(
                        conn,
                        ticket_id=fallback_id,
                        issue_key="",
                        status="new",
                        text_anonymized=fallback_text,
                        raw_payload=body,
                        trace_id=request_id,
                        autocommit=True,
                    )
                    return TicketsBatchOut(
                        accepted=1,
                        stored_file=str(raw_path),
                        reindex_started=False,
                        reindex_result={"skipped": "no serviceCall UUID; stored in tickets_inbox"},
                    )

                issue_key, text, flat = _extract_event(body)
                sc_uuid = (flat.get("sc_uuid") or "").strip()
                ticket_id = sc_uuid or issue_key.strip()
                if not ticket_id:
                    ticket_id = _safe_str(body.get("UUID") or request_id).strip() or request_id
                    raw_dump = _normalize_text(json.dumps(body, ensure_ascii=False))[:max_text_len]
                    upsert_ticket_event(
                        conn,
                        ticket_id=ticket_id,
                        issue_key="",
                        status="new",
                        text_anonymized=anonymize_text(raw_dump),
                        raw_payload=body,
                        trace_id=request_id,
                        autocommit=True,
                    )
                    return TicketsBatchOut(
                        accepted=1,
                        stored_file=str(raw_path),
                        reindex_started=False,
                        reindex_result={"skipped": "empty ticket identifier; stored raw payload only"},
                    )

                if text and len(text) > max_text_len:
                    text = text[:max_text_len]
                    flat["description_plain"] = text

                key = _state_key(flat)
                prev = _read_state(state_dir, key)
                is_dup = _same_as_last_event(prev, flat)

                _write_state(state_dir, key, _merge_state(prev, flat))
                _append_events_csv(events_file, flat)

                clean_text = anonymize_text(_normalize_text(text or ""))
                status = "closed" if _is_closed_state(flat.get("state", "")) else "new"

                upsert_ticket_event(
                    conn,
                    ticket_id=ticket_id,
                    issue_key=issue_key.strip(),
                    status=status,
                    text_anonymized=clean_text,
                    raw_payload=body,
                    trace_id=request_id,
                    autocommit=False,
                )

                if clean_text and not is_dup:
                    outbox_id = enqueue_outbox_event(
                        conn,
                        ticket_id=ticket_id,
                        payload=_build_n8n_payload(
                            ticket_id=ticket_id,
                            text_anonymized=clean_text,
                            trace_id=request_id,
                            state=flat.get("state", ""),
                            flat=flat,
                        ),
                        autocommit=False,
                    )
                    outbox_enqueued += 1
                    log.info("outbox_enqueued request_id=%s ticket_id=%s outbox_id=%s", request_id, ticket_id, outbox_id)

                conn.commit()
                accepted = 1

    except HTTPException:
        raise
    except Exception as e:
        log.exception("sd_ingest_failed request_id=%s error=%s", request_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to persist SD ticket: {e}")

    log.info(
        "sd_ingest request_id=%s accepted=%s outbox_enqueued=%s raw_file=%s",
        request_id,
        accepted,
        outbox_enqueued,
        raw_path,
    )

    return TicketsBatchOut(
        accepted=accepted,
        stored_file=str(raw_path),
        reindex_started=bool(outbox_enqueued),
        reindex_result={
            "outbox_enqueued": outbox_enqueued,
            "collection": KB_COLLECTION,
        },
    )
