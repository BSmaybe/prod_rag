# retriever/ingest_new.py
from __future__ import annotations

import csv
import glob
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

from core.vectordb import upsert_tickets, search_hits
from etl.anonymize import anonymize_text
from core.config import OLLAMA_URL, OLLAMA_MODEL
import requests

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


def _get_env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None else default


KB_JUDGE_ENABLED = _get_env_bool("KB_JUDGE_ENABLED", False)
KB_JUDGE_MODEL = _get_env_str("KB_JUDGE_MODEL", OLLAMA_MODEL)
KB_JUDGE_CACHE_DIR = _get_env_str("KB_JUDGE_CACHE_DIR", "data/tickets_clean")
KB_JUDGE_CACHE_ENABLED = _get_env_bool("KB_JUDGE_CACHE_ENABLED", False)
KB_JUDGE_MAX_CHARS = _get_env_int("KB_JUDGE_MAX_CHARS", 6000)
KB_JUDGE_TIMEOUT_SEC = float(os.getenv("KB_JUDGE_TIMEOUT_SEC", "60"))
KB_JUDGE_CONNECT_TIMEOUT_SEC = float(os.getenv("KB_JUDGE_CONNECT_TIMEOUT_SEC", "10"))
KB_JUDGE_NUM_CTX = _get_env_int("KB_JUDGE_NUM_CTX", 2048)
KB_JUDGE_NUM_PREDICT = _get_env_int("KB_JUDGE_NUM_PREDICT", 320)
KB_JUDGE_MODE = _get_env_str("KB_JUDGE_MODE", "incident").strip().lower()
KB_LOG_SKIPPED = _get_env_bool("KB_LOG_SKIPPED", True)
KB_LOG_ADDED = _get_env_bool("KB_LOG_ADDED", True)
KB_LOG_DIR = _get_env_str("KB_LOG_DIR", "data")


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_paths(digest: str) -> Tuple[Path, Path]:
    base = Path(KB_JUDGE_CACHE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    clean_path = base / f"{digest}.clean.txt"
    reject_path = base / f"{digest}.reject"
    return clean_path, reject_path


def _load_cached(clean_path: Path, reject_path: Path) -> Optional[str]:
    if reject_path.exists():
        return None
    if clean_path.exists():
        try:
            return clean_path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def _save_clean(clean_path: Path, text: str) -> None:
    try:
        clean_path.write_text(text, encoding="utf-8")
    except Exception:
        pass


def _save_reject(reject_path: Path) -> None:
    try:
        reject_path.write_text("REJECT", encoding="utf-8")
    except Exception:
        pass


def _judge_prompt(raw_text: str) -> str:
    if KB_JUDGE_MODE == "incident":
        return f"""Ты — строгий IT-аудитор базы знаний.
Твоя задача — проверить, содержит ли тикет полезную информацию для решения будущих проблем.

Сырой тикет:
\"\"\"{raw_text}\"\"\"

Правила:
1) Если НЕТ четкого описания проблемы ИЛИ НЕТ конкретных шагов решения (а только отписки), верни строго: REJECT
2) Если тикет полезен, перепиши его строго в формате:
СИМПТОМЫ: <кратко суть>
РЕШЕНИЕ: <конкретные команды, проверки, действия>
"""

    # default incident mode
    return f"""Ты — строгий IT-аудитор базы знаний.
Сырой тикет:
\"\"\"{raw_text}\"\"\"
Верни строго: REJECT или
СИМПТОМЫ: ...
РЕШЕНИЕ: ...
"""


def _parse_judge_output(text: str) -> Tuple[str, str]:
    t = _norm(text)
    if not t:
        return "", ""
    if t.strip().upper().startswith("REJECT"):
        return "", ""
    sym = ""
    sol = ""
    for line in t.splitlines():
        if line.strip().lower().startswith("симптомы"):
            sym = line.split(":", 1)[-1].strip()
        elif line.strip().lower().startswith("решение"):
            sol = line.split(":", 1)[-1].strip()
    return sym, sol


def _judge_ticket(raw_text: str) -> Optional[Tuple[str, str]]:
    prompt = _judge_prompt(raw_text[:KB_JUDGE_MAX_CHARS])
    payload = {
        "model": KB_JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": KB_JUDGE_NUM_CTX,
            "num_predict": KB_JUDGE_NUM_PREDICT,
            "stop": ["\n\nREJECT", "\n\nОтвет:", "\n\n---"],
        },
    }

    timeout = (KB_JUDGE_CONNECT_TIMEOUT_SEC, KB_JUDGE_TIMEOUT_SEC)
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"kb_judge_http_error {resp.status_code}")

    data = resp.json() if resp.content else {}
    answer = str(data.get("response", "")).strip()
    sym, sol = _parse_judge_output(answer)
    if not sym or not sol:
        return None
    return sym, sol


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
) -> Tuple[List[Dict[str, str]], Dict[str, int], List[Tuple[str, str]]]:
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
        "skipped_judge_reject": 0,
        "judge_cached": 0,
        "judge_ran": 0,
    }

    min_solution_len = _get_env_int("KB_MIN_SOLUTION_CHARS", 30)
    require_solution = _get_env_bool("KB_REQUIRE_SOLUTION", True)
    stopwords = _parse_stopwords(os.getenv("KB_STOPWORDS", ""))

    dedup_enabled = _get_env_bool("KB_DEDUP_ENABLED", True)
    dedup_score = _get_env_float("KB_DEDUP_SCORE", 0.92)
    dedup_top_k = _get_env_int("KB_DEDUP_TOP_K", 1)

    kept: List[Dict[str, str]] = []
    skipped: List[Tuple[str, str]] = []

    for r in rows:
        solution = _norm(r.get("solution_text"))

        if require_solution and not solution:
            counts["skipped_no_solution"] += 1
            skipped.append((r.get("issue_key") or "", "no_solution"))
            continue

        if solution and len(solution) < min_solution_len:
            counts["skipped_short_solution"] += 1
            skipped.append((r.get("issue_key") or "", "short_solution"))
            continue

        if solution and stopwords:
            if _normalize_for_stopwords(solution) in stopwords:
                counts["skipped_stopword"] += 1
                skipped.append((r.get("issue_key") or "", "stopword"))
                continue

        # ---- optional LLM judge (offline)
        if KB_JUDGE_ENABLED:
            raw_problem = _norm(r.get("text"))
            raw_solution = _norm(r.get("solution_text"))
            if raw_problem and raw_solution:
                # anonymize before sending to LLM
                judge_input = anonymize_text(f"{raw_problem}\n\nРешение:\n{raw_solution}")
                if KB_JUDGE_CACHE_ENABLED:
                    digest = _sha1_text(judge_input)
                    clean_path, reject_path = _cache_paths(digest)
                    cached = _load_cached(clean_path, reject_path)
                    if cached is not None:
                        counts["judge_cached"] += 1
                        sym, sol = _parse_judge_output(cached)
                        if not sym or not sol:
                            counts["skipped_judge_reject"] += 1
                            skipped.append((r.get("issue_key") or "", "judge_reject"))
                            continue
                        r["text"] = sym
                        r["solution_text"] = sol
                    elif reject_path.exists():
                        counts["skipped_judge_reject"] += 1
                        skipped.append((r.get("issue_key") or "", "judge_reject"))
                        continue
                    else:
                        try:
                            counts["judge_ran"] += 1
                            judged = _judge_ticket(judge_input)
                        except Exception:
                            judged = None
                        if not judged:
                            _save_reject(reject_path)
                            counts["skipped_judge_reject"] += 1
                            skipped.append((r.get("issue_key") or "", "judge_reject"))
                            continue
                        sym, sol = judged
                        r["text"] = sym
                        r["solution_text"] = sol
                        _save_clean(clean_path, f"СИМПТОМЫ: {sym}\nРЕШЕНИЕ: {sol}")
                else:
                    try:
                        counts["judge_ran"] += 1
                        judged = _judge_ticket(judge_input)
                    except Exception:
                        judged = None
                    if not judged:
                        counts["skipped_judge_reject"] += 1
                        skipped.append((r.get("issue_key") or "", "judge_reject"))
                        continue
                    sym, sol = judged
                    r["text"] = sym
                    r["solution_text"] = sol

        # refresh solution after possible judge rewrite
        solution = _norm(r.get("solution_text"))

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
                            skipped.append((r.get("issue_key") or "", "dedup"))
                            continue

        kept.append(r)

    counts["kept"] = len(kept)
    return kept, counts, skipped


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

    rows, counts, skipped = _filter_kb_rows(rows)
    log.info(
        "kb_filter_summary total=%s kept=%s skipped_no_solution=%s skipped_short_solution=%s "
        "skipped_stopword=%s skipped_dedup=%s skipped_judge_reject=%s judge_cached=%s judge_ran=%s",
        counts["total_rows"],
        counts["kept"],
        counts["skipped_no_solution"],
        counts["skipped_short_solution"],
        counts["skipped_stopword"],
        counts["skipped_dedup"],
        counts["skipped_judge_reject"],
        counts["judge_cached"],
        counts["judge_ran"],
    )

    if not rows:
        return 0

    # write skipped/added logs (only ticket numbers)
    log_dir = Path(KB_LOG_DIR)
    if KB_LOG_SKIPPED and skipped:
        log_dir.mkdir(parents=True, exist_ok=True)
        skipped_path = log_dir / "ingest_rejected.csv"
        with skipped_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["issue_key"])
            for key, _reason in skipped:
                w.writerow([key])

    backend = (os.getenv("INGEST_BACKEND") or os.getenv("VECTOR_BACKEND") or "qdrant").lower()

    if backend in {"qdrant", "qd"}:
        added = upsert_tickets(rows)
        if KB_LOG_ADDED and added:
            log_dir.mkdir(parents=True, exist_ok=True)
            added_path = log_dir / "ingest_added.csv"
            with added_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["issue_key"])
                for r in rows:
                    w.writerow([r.get("issue_key") or ""])
        return added

    raise RuntimeError("Unsupported INGEST_BACKEND. Use: qdrant.")
