from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.schemas.servicedesk import TicketsBatchIn
from retriever.ingest_new import ingest_new_tickets

router = APIRouter(prefix="/sd", tags=["service-desk"])


class TicketsBatchOut(BaseModel):
    accepted: int
    stored_file: str
    reindex_started: bool
    reindex_result: Optional[dict] = None


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
    """
    if not text:
        return ""
    t = text
    t = re.sub(r"<[^>]+>", " ", t)               # html tags
    t = re.sub(r"\\[a-zA-Z]+\d*", " ", t)        # rtf escapes
    t = re.sub(r"[\{\}]", " ", t)                # braces
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

@router.post("/tickets", response_model=TicketsBatchOut)
async def ingest_from_sd(request: Request):
    raw = await request.body()
    content_type = request.headers.get("content-type", "")

    # dirs
    raw_dir = Path(os.getenv("SD_RAW_DIR", "data/sd_raw"))
    events_dir = Path(os.getenv("SD_EVENTS_DIR", "data/sd_events"))
    state_dir = Path(os.getenv("SD_STATE_DIR", "data/sd_state"))
    new_tickets_dir = Path(os.getenv("NEW_TICKETS_DIR", "data/new_tickets"))

    _dump_raw(raw_dir, raw, content_type)

    body = _try_parse_json_from_anything(raw)

    max_batch = _get_env_int("SD_MAX_BATCH", 500)
    max_text_len = _get_env_int("SD_MAX_TEXT_LEN", 10000)

    new_tickets_dir.mkdir(parents=True, exist_ok=True)
    events_file = events_dir / "sd_events.csv"

    accepted = 0
    stored_file = ""

    # ===== A) batch {"tickets":[...]}
    if isinstance(body.get("tickets"), list):
        payload = TicketsBatchIn.model_validate(body)

        if not payload.tickets:
            raise HTTPException(status_code=400, detail="tickets is empty")
        if len(payload.tickets) > max_batch:
            raise HTTPException(status_code=413, detail=f"too many tickets (max {max_batch})")

        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = new_tickets_dir / f"sd_{ts_file}.csv"
        stored_file = str(out_path)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["issue_key", "text"])
            for t in payload.tickets:
                text = (t.text or "").strip()
                if not text:
                    continue
                if len(text) > max_text_len:
                    text = text[:max_text_len]
                w.writerow([t.issue_key.strip(), text])
                accepted += 1

        # events + state
        for t in payload.tickets:
            flat = {
                "sc_uuid": "",
                "issue_key": t.issue_key.strip(),
                "state": "",
                "route": "",
                "slm_service": "",
                "priority": "",
                "responsible_team": "",
                "client_employee": "",
                "client_phone": "",
                "client_email": "",
                "registration_date": "",
                "creation_date": "",
                "description_plain": (t.text or "")[:max_text_len],
            }
            _append_events_csv(events_file, flat)

            key = _state_key(flat)
            prev = _read_state(state_dir, key)
            merged = _merge_state(prev, flat)
            _write_state(state_dir, key, merged)

        if accepted == 0:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail="no valid tickets to ingest")

    # ===== B/C) Naumen event (или упакованный)
    else:
        # Скипаем "пустые" события: canceled/keyAttrChanged/newComments без message.text
        # raw уже сохранили — этого достаточно для расследований.
        if not _has_servicecall_uuid(body):
            return TicketsBatchOut(
                accepted=0,
                stored_file="skipped: no serviceCall UUID in payload",
                reindex_started=False,
                reindex_result={"skipped": "no serviceCall UUID; raw saved only"},
            )

        issue_key, text, flat = _extract_event(body)

        # доп. защита: если uuid пустой — тоже скипаем, чтобы не плодить unknown
        if (flat.get("sc_uuid") or "").strip() == "":
            return TicketsBatchOut(
                accepted=0,
                stored_file="skipped: empty sc_uuid after parse",
                reindex_started=False,
                reindex_result={"skipped": "empty sc_uuid; raw saved only"},
            )

        # ограничим текст
        if text and len(text) > max_text_len:
            text = text[:max_text_len]
            flat["description_plain"] = text

        # state всегда пишем (чтобы мерджить 5-7 событий)
        key = _state_key(flat)
        prev = _read_state(state_dir, key)
        merged = _merge_state(prev, flat)
        _write_state(state_dir, key, merged)

        # events пишем всегда (даже если текст пустой)
        _append_events_csv(events_file, flat)

        # new_tickets создаём только если есть нормальный текст и issue_key
        if _should_create_ticket_csv(issue_key, text):
            ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = new_tickets_dir / f"sd_{ts_file}.csv"
            stored_file = str(out_path)

            with out_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["issue_key", "text"])
                w.writerow([issue_key, text])

            accepted = 1
        else:
            accepted = 0
            stored_file = "no new_tickets created (event had no serviceCall description)"

    # reindex (только если реально создали new_tickets)
    reindex_started = False
    reindex_result: Optional[dict] = None
    if accepted > 0:
        try:
            reindex_started = True
            res = ingest_new_tickets()
            reindex_result = res if isinstance(res, dict) else {"result": str(res)}
        except Exception as e:
            reindex_started = False
            reindex_result = {"error": str(e)}
    else:
        reindex_result = {"skipped": "no valid ticket text; only state/events updated"}

    return TicketsBatchOut(
        accepted=accepted,
        stored_file=stored_file,
        reindex_started=reindex_started,
        reindex_result=reindex_result,
    )
