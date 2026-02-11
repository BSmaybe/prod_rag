# retriever/ingest_new.py
from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List, Optional

from core.vectordb import upsert_tickets

NEW_TICKETS_DIR = os.getenv("NEW_TICKETS_DIR", "/data/new_tickets")

# Русские названия полей из твоего CSV
COL_ISSUE = os.getenv("CSV_COL_ISSUE_KEY", "Номер запроса")
COL_TEXT = os.getenv("CSV_COL_TEXT", "Описание")
COL_SOLUTION = os.getenv("CSV_COL_SOLUTION", "Описание решения")
COL_SERVICE = os.getenv("CSV_COL_SERVICE", "Услуга")

CSV_DELIMITER = os.getenv("CSV_DELIMITER", ";")
CSV_ENCODING = os.getenv("CSV_ENCODING", "utf-8")


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

    backend = (os.getenv("INGEST_BACKEND") or os.getenv("VECTOR_BACKEND") or "qdrant").lower()

    if backend in {"qdrant", "qd"}:
        return upsert_tickets(rows)

    raise RuntimeError("Unsupported INGEST_BACKEND. Use: qdrant.")
