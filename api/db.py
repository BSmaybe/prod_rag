from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

log = logging.getLogger("rag.db")

DB_CONNECT_MAX_RETRIES = int(os.getenv("DB_CONNECT_MAX_RETRIES", "20"))
DB_CONNECT_RETRY_DELAY_SEC = float(os.getenv("DB_CONNECT_RETRY_DELAY_SEC", "2"))


def get_db_url() -> str:
    db_url = os.getenv("DB_URL", "").strip()
    if not db_url:
        raise RuntimeError("DB_URL is required for tickets_inbox journal")
    return db_url


@contextmanager
def db_conn() -> Generator[psycopg.Connection, None, None]:
    last_error: Exception | None = None
    conn: psycopg.Connection | None = None
    for attempt in range(1, DB_CONNECT_MAX_RETRIES + 1):
        try:
            conn = psycopg.connect(get_db_url())
            break
        except Exception as e:
            last_error = e
            if attempt >= DB_CONNECT_MAX_RETRIES:
                raise
            log.warning(
                "db_connect_retry attempt=%s/%s delay_sec=%.1f error=%s",
                attempt,
                DB_CONNECT_MAX_RETRIES,
                DB_CONNECT_RETRY_DELAY_SEC,
                e,
            )
            time.sleep(DB_CONNECT_RETRY_DELAY_SEC)
    if conn is None:
        raise RuntimeError(f"Unable to connect to DB after retries: {last_error}")
    try:
        yield conn
    finally:
        conn.close()


def _payload_hash(payload: dict[str, Any]) -> str:
    """
    Стабильный хэш для дедупа.
    ВАЖНО: исключаем trace_id и любые "плавающие" поля, чтобы дубль не создавал новую запись.
    """
    if not isinstance(payload, dict):
        payload = {"value": payload}

    normalized = dict(payload)

    # trace_id меняется каждый раз — выкидываем из хэша
    normalized.pop("trace_id", None)

    # иногда ещё может гулять request_id и т.п. — если добавишь в payload, тоже выкидывай тут
    normalized.pop("request_id", None)

    s = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets_inbox (
                ticket_id text PRIMARY KEY,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                status text NOT NULL,
                text_anonymized text NOT NULL,
                raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
                trace_id text,
                last_comment_sent timestamptz NULL
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tickets_inbox_status
            ON tickets_inbox(status);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tickets_inbox_updated_at
            ON tickets_inbox(updated_at DESC);
            """
        )
        cur.execute("ALTER TABLE tickets_inbox ADD COLUMN IF NOT EXISTS issue_key text;")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tickets_inbox_issue_key
            ON tickets_inbox(issue_key);
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS n8n_outbox (
                id bigserial PRIMARY KEY,
                ticket_id text NOT NULL,
                payload jsonb NOT NULL,
                payload_hash text NULL,
                status text NOT NULL DEFAULT 'pending',
                attempts int NOT NULL DEFAULT 0,
                next_retry_at timestamptz NOT NULL DEFAULT now(),
                last_error text NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )

        # если таблица существовала раньше без payload_hash — добавим
        cur.execute("ALTER TABLE n8n_outbox ADD COLUMN IF NOT EXISTS payload_hash text;")

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_n8n_outbox_status_next_retry
            ON n8n_outbox(status, next_retry_at);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_n8n_outbox_ticket_id
            ON n8n_outbox(ticket_id);
            """
        )

        # ДЕДУП: один и тот же payload_hash для ticket_id может быть только один раз
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_n8n_outbox_ticket_payload_hash
            ON n8n_outbox(ticket_id, payload_hash);
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ticket_jobs (
                job_id uuid PRIMARY KEY,
                ticket_id text NOT NULL,
                text_anonymized text NOT NULL,
                status text NOT NULL DEFAULT 'pending',
                result_payload jsonb NULL,
                error text NULL,
                attempts int NOT NULL DEFAULT 0,
                created_at timestamptz NOT NULL DEFAULT now(),
                started_at timestamptz NULL,
                finished_at timestamptz NULL,
                updated_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ticket_jobs_status_created
            ON ticket_jobs(status, created_at);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ticket_jobs_ticket_created
            ON ticket_jobs(ticket_id, created_at DESC);
            """
        )

    conn.commit()


def upsert_ticket_event(
    conn: psycopg.Connection,
    *,
    ticket_id: str,
    issue_key: str | None = None,
    status: str,
    text_anonymized: str,
    raw_payload: dict[str, Any],
    trace_id: str | None = None,
    autocommit: bool = True,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tickets_inbox (
                ticket_id, issue_key, status, text_anonymized, raw_payload, trace_id
            ) VALUES (
                %s, %s, %s, %s, %s::jsonb, %s
            )
            ON CONFLICT (ticket_id)
            DO UPDATE SET
                updated_at = now(),
                issue_key = CASE
                    WHEN COALESCE(NULLIF(EXCLUDED.issue_key, ''), '') <> '' THEN EXCLUDED.issue_key
                    ELSE tickets_inbox.issue_key
                END,
                status = EXCLUDED.status,
                text_anonymized = EXCLUDED.text_anonymized,
                raw_payload = EXCLUDED.raw_payload,
                trace_id = EXCLUDED.trace_id;
            """,
            (
                ticket_id,
                issue_key,
                status,
                text_anonymized,
                json.dumps(raw_payload, ensure_ascii=False),
                trace_id,
            ),
        )
    if autocommit:
        conn.commit()


def set_ticket_status(
    conn: psycopg.Connection,
    *,
    ticket_id: str,
    status: str,
    mark_comment_sent: bool = False,
    autocommit: bool = True,
) -> None:
    sql = """
    UPDATE tickets_inbox
    SET status = %s,
        updated_at = now(),
        last_comment_sent = CASE
            WHEN %s THEN now()
            ELSE last_comment_sent
        END
    WHERE ticket_id = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (status, mark_comment_sent, ticket_id))
    if autocommit:
        conn.commit()


def enqueue_outbox_event(
    conn: psycopg.Connection,
    *,
    ticket_id: str,
    payload: dict[str, Any],
    status: str = "pending",
    autocommit: bool = True,
) -> int:
    """
    Идемпотентная вставка: если такой же payload_hash уже был — не вставляем второй раз.
    Возвращает:
      - id новой записи, если вставили
      - 0, если это дубль (уже есть)
    """
    ph = _payload_hash(payload)

    outbox_id = 0
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO n8n_outbox (ticket_id, payload, payload_hash, status)
            VALUES (%s, %s::jsonb, %s, %s)
            ON CONFLICT (ticket_id, payload_hash) DO NOTHING
            RETURNING id;
            """,
            (ticket_id, json.dumps(payload, ensure_ascii=False), ph, status),
        )
        row = cur.fetchone()
        outbox_id = int(row[0]) if row else 0

    if autocommit:
        conn.commit()

    return outbox_id


def claim_outbox_batch(
    conn: psycopg.Connection,
    *,
    limit: int,
    autocommit: bool = True,
) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            WITH picked AS (
                SELECT id
                FROM n8n_outbox
                WHERE status IN ('pending', 'retry')
                  AND next_retry_at <= now()
                ORDER BY next_retry_at ASC, id ASC
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            UPDATE n8n_outbox o
            SET status = 'processing',
                updated_at = now()
            FROM picked
            WHERE o.id = picked.id
            RETURNING o.id, o.ticket_id, o.payload, o.payload_hash, o.status, o.attempts,
                      o.next_retry_at, o.last_error, o.created_at, o.updated_at;
            """,
            (limit,),
        )
        rows = cur.fetchall() or []
    if autocommit:
        conn.commit()
    return [dict(r) for r in rows]


def mark_outbox_sent(
    conn: psycopg.Connection,
    *,
    outbox_id: int,
    autocommit: bool = True,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE n8n_outbox
            SET status = 'sent',
                updated_at = now(),
                last_error = NULL
            WHERE id = %s;
            """,
            (outbox_id,),
        )
    if autocommit:
        conn.commit()


def mark_outbox_retry(
    conn: psycopg.Connection,
    *,
    outbox_id: int,
    attempts: int,
    last_error: str,
    next_retry_at: datetime,
    autocommit: bool = True,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE n8n_outbox
            SET status = 'retry',
                attempts = %s,
                last_error = %s,
                next_retry_at = %s,
                updated_at = now()
            WHERE id = %s;
            """,
            (attempts, last_error[:2000], next_retry_at, outbox_id),
        )
    if autocommit:
        conn.commit()


def mark_outbox_dead(
    conn: psycopg.Connection,
    *,
    outbox_id: int,
    last_error: str,
    autocommit: bool = True,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE n8n_outbox
            SET status = 'dead_letter',
                last_error = %s,
                updated_at = now()
            WHERE id = %s;
            """,
            (last_error[:2000], outbox_id),
        )
    if autocommit:
        conn.commit()


def enqueue_ticket_job(
    conn: psycopg.Connection,
    *,
    ticket_id: str,
    text_anonymized: str,
    autocommit: bool = True,
) -> str:
    job_id = uuid4()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ticket_jobs (job_id, ticket_id, text_anonymized, status)
            VALUES (%s, %s, %s, 'pending');
            """,
            (str(job_id), ticket_id, text_anonymized),
        )
    if autocommit:
        conn.commit()
    return str(job_id)


def claim_ticket_jobs_batch(
    conn: psycopg.Connection,
    *,
    limit: int,
    autocommit: bool = True,
) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            WITH picked AS (
                SELECT job_id
                FROM ticket_jobs
                WHERE status = 'pending'
                ORDER BY created_at ASC, job_id ASC
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            UPDATE ticket_jobs j
            SET status = 'running',
                attempts = j.attempts + 1,
                started_at = COALESCE(j.started_at, now()),
                updated_at = now()
            FROM picked
            WHERE j.job_id = picked.job_id
            RETURNING j.job_id, j.ticket_id, j.text_anonymized, j.status, j.result_payload,
                      j.error, j.attempts, j.created_at, j.started_at, j.finished_at, j.updated_at;
            """,
            (limit,),
        )
        rows = cur.fetchall() or []
    if autocommit:
        conn.commit()
    return [dict(r) for r in rows]


def mark_ticket_job_running(
    conn: psycopg.Connection,
    *,
    job_id: str,
    autocommit: bool = True,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ticket_jobs
            SET status = 'running',
                attempts = attempts + 1,
                started_at = COALESCE(started_at, now()),
                updated_at = now()
            WHERE job_id = %s;
            """,
            (job_id,),
        )
    if autocommit:
        conn.commit()


def mark_ticket_job_done(
    conn: psycopg.Connection,
    *,
    job_id: str,
    result_payload: dict[str, Any],
    autocommit: bool = True,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ticket_jobs
            SET status = 'done',
                result_payload = %s::jsonb,
                error = NULL,
                finished_at = now(),
                updated_at = now()
            WHERE job_id = %s;
            """,
            (json.dumps(result_payload, ensure_ascii=False), job_id),
        )
    if autocommit:
        conn.commit()


def mark_ticket_job_error(
    conn: psycopg.Connection,
    *,
    job_id: str,
    error: str,
    autocommit: bool = True,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ticket_jobs
            SET status = 'error',
                error = %s,
                finished_at = now(),
                updated_at = now()
            WHERE job_id = %s;
            """,
            (error[:4000], job_id),
        )
    if autocommit:
        conn.commit()


def get_ticket_job(conn: psycopg.Connection, job_id: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT job_id, ticket_id, text_anonymized, status, result_payload, error,
                   attempts, created_at, started_at, finished_at, updated_at
            FROM ticket_jobs
            WHERE job_id = %s
            """,
            (job_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def get_ticket(conn: psycopg.Connection, ticket_id: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ticket_id, issue_key, status, text_anonymized, raw_payload, trace_id,
                   created_at, updated_at, last_comment_sent
            FROM tickets_inbox
            WHERE ticket_id = %s
            """,
            (ticket_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def get_ticket_issue_key(conn: psycopg.Connection, ticket_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT issue_key
            FROM tickets_inbox
            WHERE ticket_id = %s
            """,
            (ticket_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    issue_key = row[0]
    if issue_key is None:
        return None
    issue_key_str = str(issue_key).strip()
    return issue_key_str or None
