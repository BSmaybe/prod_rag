from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Generator

import psycopg
from psycopg.rows import dict_row


def get_db_url() -> str:
    db_url = os.getenv("DB_URL", "").strip()
    if not db_url:
        raise RuntimeError("DB_URL is required for tickets_inbox journal")
    return db_url


@contextmanager
def db_conn() -> Generator[psycopg.Connection, None, None]:
    conn = psycopg.connect(get_db_url())
    try:
        yield conn
    finally:
        conn.close()


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
    conn.commit()


def upsert_ticket_event(
    conn: psycopg.Connection,
    *,
    ticket_id: str,
    status: str,
    text_anonymized: str,
    raw_payload: dict[str, Any],
    trace_id: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tickets_inbox (
                ticket_id, status, text_anonymized, raw_payload, trace_id
            ) VALUES (
                %s, %s, %s, %s::jsonb, %s
            )
            ON CONFLICT (ticket_id)
            DO UPDATE SET
                updated_at = now(),
                status = EXCLUDED.status,
                text_anonymized = EXCLUDED.text_anonymized,
                raw_payload = EXCLUDED.raw_payload,
                trace_id = EXCLUDED.trace_id;
            """,
            (
                ticket_id,
                status,
                text_anonymized,
                json.dumps(raw_payload, ensure_ascii=False),
                trace_id,
            ),
        )
    conn.commit()


def set_ticket_status(
    conn: psycopg.Connection,
    *,
    ticket_id: str,
    status: str,
    mark_comment_sent: bool = False,
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
    conn.commit()


def get_ticket(conn: psycopg.Connection, ticket_id: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ticket_id, status, text_anonymized, raw_payload, trace_id,
                   created_at, updated_at, last_comment_sent
            FROM tickets_inbox
            WHERE ticket_id = %s
            """,
            (ticket_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None
