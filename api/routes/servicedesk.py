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


def _strip_rtf(text: str) -> str:
    """
    Упрощённая очистка:
    - html теги
    - rtf-эскейпы вида \par \b0 \u1234 ...
    - декодирование HTML entities (&nbsp; и т.п.)
    """
    if not text:
        return ""
    t = text

    # 0) decode HTML entities early
    t = html.unescape(t)  # <-- добавь это

    t = re.sub(r"<[^>]+>", " ", t)               # html tags
    t = re.sub(r"\\[a-zA-Z]+\d*", " ", t)        # rtf escapes
    t = re.sub(r"[\{\}]", " ", t)                # braces

    # на всякий: после unescape nbsp может стать \xa0
    t = t.replace("\xa0", " ")

    return re.sub(r"\s+", " ", t).strip()


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
        encoding="utf-8"
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

    Мы должны:
    1) Распарсить внешний JSON
    2) Если есть message.text — распарсить JSON из него
    3) Иначе пробовать старые варианты
    """
    s = raw_bytes.decode("utf-8", errors="replace").strip()

    # 1) пробуем внешний JSON
    try:
        outer = json.loads(s)
    except Exception:
        outer = None

    if isinstance(outer, dict):
        # 2) ГЛАВНОЕ — message.text
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

        # 3) старый вариант — вдруг text на верхнем уровне
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

        # 4) если это batch {"tickets":[...]}
        return outer

    # 5) fallback — JSON внутри строки
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
    """
    True если payload содержит хотя бы какой-то serviceCall.UUID
    (в serviceCall или header.serviceCall)
    """
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
    Пытаемся собрать максимальный плоский event из serviceCall.
    Возвращаем (issue_key, text, flat)
    """
    hdr_sc = _get_in(body, ["header", "serviceCall"], default={})
    sc = body.get("serviceCall") or {}

    sc_uuid = _safe_str(sc.get("UUID") or (hdr_sc.get("UUID") if isinstance(hdr_sc, dict) else "") or "")
    issue_key = _safe_str((hdr_sc.get("title") if isinstance(hdr_sc, dict) else None) or sc.get("title") or sc_uuid or "unknown")
    state = _safe_str(sc.get("state") or (hdr_sc.get("state") if isinstance(hdr_sc, dict) else "") or "")

    # описание может быть в разных местах
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

    # totalValue разворачиваем в tv_*
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

    # Дедуп: если последняя строка по сути такая же — не пишем новую
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

    # читаем существующие колонки
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_cols = next(reader, [])

    new_cols = list(existing_cols)
    for k in row.keys():
        if k not in new_cols:
            new_cols.append(k)

    # если колонки расширились — перепишем файл с новым header
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
    """
    Для merge по тикету.
    Предпочтительно serviceCall UUID, иначе issue_key.
    """
    sc_uuid = (flat.get("sc_uuid") or "").strip()
    if sc_uuid:
        return sc_uuid
    issue_key = (flat.get("issue_key") or "").strip()
    return issue_key or "unknown"


def _merge_state(prev: Dict[str, Any], flat: Dict[str, str]) -> Dict[str, Any]:
    """
    Апдейт состояния тикета:
    - не затирать непустые значения пустыми
    - events хранить списком (последние N)
    - дедуп одинаковых подряд событий
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = dict(prev or {})
    out.setdefault("updated_at", now)
    out.setdefault("created_at", now)
    out["updated_at"] = now

    # поля
    for k, v in flat.items():
        v = _safe_str(v).strip()
        if v:
            out[k] = v
        else:
            out.setdefault(k, "")

    # события (последние 50) + дедуп по последней записи
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
    safe = re.sub(r"[^a-zA-Z0-9_\-$\.]", "_", state_key)
    if not safe:
        safe = "unknown"
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


def _should_create_ticket_csv(issue_key: str, text: str) -> bool:
    """
    Создаём new_tickets CSV только если:
    - issue_key не unknown
    - есть непустой текст
    """
    if not issue_key or issue_key.strip().lower() == "unknown":
        return False
    if not text or not text.strip():
        return False
    return True


# =========================
# Route
# =========================

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


def _extract_solution_text(body: Dict[str, Any]) -> str:
    sc = body.get("serviceCall") if isinstance(body, dict) else {}
    if not isinstance(sc, dict):
        sc = {}

    candidates: List[str] = []

    key_candidates = [
        "solution",
        "resolution",
        "result",
        "closeComment",
        "decision",
        "workaround",
        "techDesc",
        "descriptionInRTF",
        "description",
    ]
    for key in key_candidates:
        v = sc.get(key)
        if isinstance(v, str) and v.strip():
            candidates.append(_strip_rtf(v))

    tv = sc.get("totalValue")
    if isinstance(tv, list):
        for item in tv:
            if not isinstance(item, dict):
                continue
            title = _safe_str(item.get("title")).strip().lower()
            text_value = item.get("textValue")
            if text_value is None:
                text_value = item.get("value")
            if not isinstance(text_value, str):
                continue
            if any(token in title for token in ["реш", "причин", "cause", "root", "диагност", "fix"]):
                candidates.append(_strip_rtf(text_value))

    merged = "\n\n".join([c for c in candidates if c.strip()]).strip()
    return merged


def _looks_like_low_quality_solution(text: str) -> bool:
    normalized = text.lower()
    bad_patterns = [
        r"\bсамо\s+прошл",
        r"\bпереустанов",
        r"\bпочистил[аи]?\s+кэш\b",
        r"\bне\s+воспроизвел",
        r"\bдубликат\b",
        r"\bзакрыт[оа]?\b",
    ]
    return any(re.search(rx, normalized) for rx in bad_patterns)


def _solution_quality_ok(solution_text: str, flat: Dict[str, str]) -> bool:
    text = (solution_text or "").strip()
    if len(text) < 300:
        return False
    if _looks_like_low_quality_solution(text):
        return False

    normalized = text.lower()
    feature_count = 0

    if re.search(r"(error|ошиб|exception|trace|stack|код)", normalized):
        feature_count += 1
    if re.search(r"(шаг|воспроизв|проверк|диагност|лог|скрин)", normalized):
        feature_count += 1
    if (flat.get("route") or flat.get("slm_service") or flat.get("responsible_team")):
        feature_count += 1
    if re.search(r"(причин|корен|изменен|изменён|change|deploy|релиз|конфиг)", normalized):
        feature_count += 1

    return feature_count >= 2


def _build_n8n_payload(
    *,
    ticket_id: str,
    text_anonymized: str,
    trace_id: str,
    state: str,
    flat: Dict[str, str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ticket_id": ticket_id,
        "text": text_anonymized,
        "trace_id": trace_id,
        "state": state,
        "is_closed": _is_closed_state(state),
    }
    if flat.get("route"):
        payload["service"] = flat.get("route")
    if flat.get("slm_service"):
        payload["component"] = flat.get("slm_service")
    return payload


def _fetch_solution_from_naumen(ticket_id: str) -> str:
    base_url = (os.getenv("NAUMEN_API_URL") or "").strip()
    if not base_url:
        return ""

    if "{ticket_id}" in base_url:
        url = base_url.format(ticket_id=ticket_id)
    elif base_url.endswith("/"):
        url = f"{base_url}{ticket_id}"
    else:
        url = f"{base_url}/{ticket_id}"

    headers = {"Accept": "application/json"}
    api_key = (os.getenv("SERVICE_API_KEY") or os.getenv("SERVICE_DESK_API_KEY") or "").strip()
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return ""

    if not isinstance(data, dict):
        return ""

    candidates = []
    for key in ["solution", "resolution", "result", "closeComment", "techDesc", "description", "descriptionInRTF"]:
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            candidates.append(_strip_rtf(v))

    sc = data.get("serviceCall")
    if isinstance(sc, dict):
        for key in ["solution", "resolution", "result", "closeComment", "techDesc", "description", "descriptionInRTF"]:
            v = sc.get(key)
            if isinstance(v, str) and v.strip():
                candidates.append(_strip_rtf(v))

    return "\n\n".join(candidates).strip()


async def _index_closed_ticket_if_quality(
    *,
    ticket_id: str,
    body: Dict[str, Any],
    flat: Dict[str, str],
    request_id: str,
) -> None:
    solution_text = _extract_solution_text(body)
    if not solution_text:
        solution_text = _fetch_solution_from_naumen(ticket_id)
    if not solution_text:
        log.info("kb_skip request_id=%s ticket_id=%s reason=no_solution_text", request_id, ticket_id)
        return

    if not _solution_quality_ok(solution_text, flat):
        log.info("kb_skip request_id=%s ticket_id=%s reason=quality_filter", request_id, ticket_id)
        return

    record = {
        "issue_key": ticket_id,
        "text": solution_text,
        "state": flat.get("state", ""),
        "route": flat.get("route", ""),
        "slm_service": flat.get("slm_service", ""),
    }
    try:
        added = await asyncio.to_thread(upsert_tickets, [record])
        log.info(
            "kb_upserted request_id=%s ticket_id=%s collection=%s points=%s",
            request_id,
            ticket_id,
            KB_COLLECTION,
            added,
        )
    except Exception as e:
        log.warning("kb_upsert_failed request_id=%s ticket_id=%s error=%s", request_id, ticket_id, e)


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

    kb_tasks: List[asyncio.Task[Any]] = []
    outbox_enqueued = 0
    accepted = 0

    try:
        with db_conn() as conn:
            ensure_schema(conn)

            if isinstance(body.get("tickets"), list):
                payload = TicketsBatchIn.model_validate(body)
                if not payload.tickets:
                    raise HTTPException(status_code=400, detail="tickets is empty")
                if len(payload.tickets) > max_batch:
                    raise HTTPException(status_code=413, detail=f"too many tickets (max {max_batch})")

                for t in payload.tickets:
                    ticket_id = t.issue_key.strip()
                    raw_text = (t.text or "").strip()
                    if not ticket_id or not raw_text:
                        continue
                    clean_text = anonymize_text(raw_text[:max_text_len])
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
                    _append_events_csv(events_file, flat)
                    key = _state_key(flat)
                    prev = _read_state(state_dir, key)
                    _write_state(state_dir, key, _merge_state(prev, flat))
                    upsert_ticket_event(
                        conn,
                        ticket_id=ticket_id,
                        status="new",
                        text_anonymized=clean_text,
                        raw_payload={"tickets_item": t.model_dump(), "parsed": body},
                        trace_id=request_id,
                        autocommit=False,
                    )
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
                    conn.commit()
                    outbox_enqueued += 1
                    log.info(
                        "outbox_enqueued request_id=%s ticket_id=%s outbox_id=%s",
                        request_id,
                        ticket_id,
                        outbox_id,
                    )
                    accepted += 1
            else:
                if not _has_servicecall_uuid(body):
                    fallback_id = _safe_str(body.get("UUID") or body.get("title") or request_id).strip() or request_id
                    fallback_text = anonymize_text(json.dumps(body, ensure_ascii=False))[:max_text_len]
                    upsert_ticket_event(
                        conn,
                        ticket_id=fallback_id,
                        status="new",
                        text_anonymized=fallback_text,
                        raw_payload=body,
                        trace_id=request_id,
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
                    upsert_ticket_event(
                        conn,
                        ticket_id=ticket_id,
                        status="new",
                        text_anonymized=anonymize_text(json.dumps(body, ensure_ascii=False))[:max_text_len],
                        raw_payload=body,
                        trace_id=request_id,
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
                _write_state(state_dir, key, _merge_state(prev, flat))
                _append_events_csv(events_file, flat)

                clean_text = anonymize_text(text or "")
                status = "closed" if _is_closed_state(flat.get("state", "")) else "new"
                upsert_ticket_event(
                    conn,
                    ticket_id=ticket_id,
                    status=status,
                    text_anonymized=clean_text,
                    raw_payload=body,
                    trace_id=request_id,
                    autocommit=False,
                )

                if clean_text:
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
                    log.info(
                        "outbox_enqueued request_id=%s ticket_id=%s outbox_id=%s",
                        request_id,
                        ticket_id,
                        outbox_id,
                    )
                conn.commit()

                if _is_closed_state(flat.get("state", "")):
                    kb_tasks.append(
                        asyncio.create_task(
                            _index_closed_ticket_if_quality(
                                ticket_id=ticket_id,
                                body=body,
                                flat=flat,
                                request_id=request_id,
                            )
                        )
                    )
                accepted = 1
    except HTTPException:
        raise
    except Exception as e:
        log.exception("sd_ingest_failed request_id=%s error=%s", request_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to persist SD ticket: {e}")

    log.info(
        "sd_ingest request_id=%s accepted=%s outbox_enqueued=%s kb_tasks=%s raw_file=%s",
        request_id,
        accepted,
        outbox_enqueued,
        len(kb_tasks),
        raw_path,
    )

    return TicketsBatchOut(
        accepted=accepted,
        stored_file=str(raw_path),
        reindex_started=bool(outbox_enqueued or kb_tasks),
        reindex_result={
            "outbox_enqueued": outbox_enqueued,
            "kb_tasks": len(kb_tasks),
            "collection": KB_COLLECTION,
        },
    )
