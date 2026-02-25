# retriever/ingest_new.py
from __future__ import annotations

import csv
import glob
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Callable, Any

from core.vectordb import upsert_tickets, search_hits
from etl.anonymize import anonymize_text

NEW_TICKETS_DIR = os.getenv("NEW_TICKETS_DIR", "/data/new_tickets")

# Русские названия полей из твоего CSV
COL_ISSUE = os.getenv("CSV_COL_ISSUE_KEY", "Номер запроса")
COL_TEXT = os.getenv("CSV_COL_TEXT", "Описание")
COL_SOLUTION = os.getenv("CSV_COL_SOLUTION", "Описание решения")
COL_SERVICE = os.getenv("CSV_COL_SERVICE", "Услуга")

CSV_DELIMITER = os.getenv("CSV_DELIMITER", ";")
CSV_ENCODING = os.getenv("CSV_ENCODING", "utf-8")

log = logging.getLogger("rag.ingest")

_STOPWORD_SPLIT_RE = re.compile(r"\s*,\s*")
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _make_issue_key(raw: str) -> str:
    raw = _norm(raw)
    if not raw:
        return ""
    # чтобы было похоже на твою схему RPxxxx
    if raw.upper().startswith("RP"):
        return raw
    # если в CSV просто число типа 5048476
    if raw.isdigit():
        return f"RP{raw}"
    return raw


def _detect_delimiter(path: str) -> str:
    # на всякий случай авто-детект, если delimiter внезапно не ;
    try:
        with open(path, "r", encoding=CSV_ENCODING, errors="replace", newline="") as f:
            sample = f.read(4096)
        if sample.count(";") >= sample.count(","):
            return ";"
        return ","
    except Exception:
        return CSV_DELIMITER


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_for_stopwords(text: str) -> str:
    t = _norm(text).lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _parse_stopwords(raw: str) -> set[str]:
    if not raw:
        return set()
    parts = _STOPWORD_SPLIT_RE.split(raw)
    return {s for s in (_normalize_for_stopwords(p) for p in parts) if s}


def _build_dedup_text(problem_text: str, solution_text: str, service: str) -> str:
    parts: List[str] = []
    if service:
        parts.append(f"[SERVICE] {service}")
    if problem_text:
        parts.append(problem_text)
    if solution_text:
        parts.append(f"[SOLUTION] {solution_text}")
    return "\n".join(parts).strip()


def _filter_kb_rows(
    rows: List[Dict[str, str]],
    *,
    search_fn: Callable[[str, int], List[Dict[str, Any]]] = search_hits,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Возвращает (отфильтрованные строки, счётчики).
    Фильтры применяются ТОЛЬКО для bulk reindex.
    """
    counts = {
        "total_rows": len(rows),
        "kept": 0,
        "skipped_no_solution": 0,
        "skipped_short_solution": 0,
        "skipped_stopword": 0,
        "skipped_dedup": 0,
    }

    min_solution_len = _get_env_int("KB_MIN_SOLUTION_CHARS", 30)
    require_solution = _get_env_bool("KB_REQUIRE_SOLUTION", True)
    stopwords = _parse_stopwords(os.getenv("KB_STOPWORDS", ""))

    dedup_enabled = _get_env_bool("KB_DEDUP_ENABLED", True)
    dedup_score = _get_env_float("KB_DEDUP_SCORE", 0.92)
    dedup_top_k = _get_env_int("KB_DEDUP_TOP_K", 1)

    kept: List[Dict[str, str]] = []

    for r in rows:
        solution = _norm(r.get("solution_text"))

        if require_solution and not solution:
            counts["skipped_no_solution"] += 1
            continue

        if solution and len(solution) < min_solution_len:
            counts["skipped_short_solution"] += 1
            continue

        if solution and stopwords:
            if _normalize_for_stopwords(solution) in stopwords:
                counts["skipped_stopword"] += 1
                continue

        if dedup_enabled:
            raw_problem = _norm(r.get("text"))
            if raw_problem:
                # anonymize before search to match KB content
                problem_text = anonymize_text(raw_problem)
                solution_text = anonymize_text(solution)
                service = anonymize_text(_norm(r.get("service")))
                dedup_text = _build_dedup_text(problem_text, solution_text, service)
                if dedup_text:
                    try:
                        hits = search_fn(dedup_text, top_k=dedup_top_k)
                    except Exception:
                        hits = []
                    if hits:
                        top_score = float(hits[0].get("score") or 0.0)
                        if top_score >= dedup_score:
                            counts["skipped_dedup"] += 1
                            continue

        kept.append(r)

    counts["kept"] = len(kept)
    return kept, counts


def _load_csv_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    if not os.path.isdir(NEW_TICKETS_DIR):
        return rows

    pattern = os.path.join(NEW_TICKETS_DIR, "*.csv")
    for path in sorted(glob.glob(pattern)):
        delim = _detect_delimiter(path)

        with open(path, newline="", encoding=CSV_ENCODING, errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delim)

            # если вдруг заголовки “склеились” — значит delimiter неверный
            if reader.fieldnames and len(reader.fieldnames) == 1 and ";" in (reader.fieldnames[0] or ""):
                # принудительно переключаем на ;
                f.seek(0)
                reader = csv.DictReader(f, delimiter=";")

            for r in reader:
                issue_key = _make_issue_key(r.get(COL_ISSUE))
                text = _norm(r.get(COL_TEXT))

                if not issue_key or not text:
                    continue

                solution_text = _norm(r.get(COL_SOLUTION))
                service = _norm(r.get(COL_SERVICE))

                row: Dict[str, str] = {"issue_key": issue_key, "text": text}
                if solution_text:
                    row["solution_text"] = solution_text
                if service:
                    row["service"] = service

                rows.append(row)

    return rows


def ingest_new_tickets() -> int | dict[str, int]:
    rows = _load_csv_rows()
    if not rows:
        return 0

    rows, counts = _filter_kb_rows(rows)
    log.info(
        "kb_filter_summary total=%s kept=%s skipped_no_solution=%s skipped_short_solution=%s "
        "skipped_stopword=%s skipped_dedup=%s",
        counts["total_rows"],
        counts["kept"],
        counts["skipped_no_solution"],
        counts["skipped_short_solution"],
        counts["skipped_stopword"],
        counts["skipped_dedup"],
    )

    if not rows:
        return 0

    backend = (os.getenv("INGEST_BACKEND") or os.getenv("VECTOR_BACKEND") or "qdrant").lower()

    if backend in {"qdrant", "qd"}:
        return upsert_tickets(rows)

    raise RuntimeError("Unsupported INGEST_BACKEND. Use: qdrant.")
