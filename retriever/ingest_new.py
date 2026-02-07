# retriever/ingest_new.py
from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List

from core.vectordb import upsert_tickets


# Путь внутри контейнера (см. volume в docker-compose)
NEW_TICKETS_DIR = os.getenv("NEW_TICKETS_DIR", "/data/new_tickets")


def _load_csv_rows() -> List[Dict[str, str]]:
    """
    Читает все *.csv из NEW_TICKETS_DIR.
    Поддерживает столбцы:
      - issue_key (required)
      - text (required)
      - solution_text (optional)
    """
    rows: List[Dict[str, str]] = []

    if not os.path.isdir(NEW_TICKETS_DIR):
        return rows

    pattern = os.path.join(NEW_TICKETS_DIR, "*.csv")
    for path in glob.glob(pattern):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                issue_key = (r.get("issue_key") or "").strip()
                text = (r.get("text") or "").strip()
                if not issue_key or not text:
                    continue
                solution_text = (r.get("solution_text") or "").strip()
                row: Dict[str, str] = {"issue_key": issue_key, "text": text}
                if solution_text:
                    row["solution_text"] = solution_text
                rows.append(row)

    return rows


def _ingest_pgvector_legacy(rows: List[Dict[str, str]]) -> int:
    """
    Legacy fallback для pgvector. По умолчанию не используется.
    """
    import psycopg

    from retriever.hybrid_search import TABLE, ensure_schema, get_db_url, get_model

    model = get_model()
    texts = [r["text"] for r in rows]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    with psycopg.connect(get_db_url()) as conn, conn.cursor() as cur:
        ensure_schema(conn)

        cur.execute(f"SELECT DISTINCT issue_key FROM {TABLE}")
        existing_keys = {row[0] for row in cur.fetchall()}

        to_insert = []
        for r, emb in zip(rows, embeddings):
            if r["issue_key"] in existing_keys:
                continue
            to_insert.append((r["issue_key"], r["text"], emb))

        if not to_insert:
            return 0

        sql = f"""
        INSERT INTO {TABLE} (issue_key, text_chunk, embedding)
        VALUES (%s, %s, %s)
        """
        for issue_key, text, emb in to_insert:
            cur.execute(sql, (issue_key, text, emb))

        conn.commit()
        return len(to_insert)


def ingest_new_tickets() -> int | dict[str, int]:
    """
    Активный контур: пишет новые тикеты в Qdrant.
    Legacy (по запросу): может писать в pgvector при INGEST_BACKEND=pgvector.
    """
    rows = _load_csv_rows()
    if not rows:
        return 0

    backend = (os.getenv("INGEST_BACKEND") or os.getenv("VECTOR_BACKEND") or "qdrant").lower()

    if backend in {"qdrant", "qd"}:
        return upsert_tickets(rows)

    if backend in {"pgvector", "postgres", "pg"}:
        return _ingest_pgvector_legacy(rows)

    if backend == "both":
        return {
            "qdrant": upsert_tickets(rows),
            "pgvector": _ingest_pgvector_legacy(rows),
        }

    raise RuntimeError(
        "Unsupported INGEST_BACKEND. Use one of: qdrant, pgvector, both."
    )
