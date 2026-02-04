# api/feedback.py
from __future__ import annotations

from typing import List, Optional

import psycopg
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from retriever.hybrid_search import get_db_url  # используем тот же DSN


router = APIRouter(prefix="/feedback", tags=["feedback"])


def ensure_feedback_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
              id              bigserial PRIMARY KEY,
              query           text NOT NULL,
              answer_full_text text NOT NULL,
              is_helpful      boolean NOT NULL,
              comment         text,
              used_issue_keys text[]
            );
            """
        )


class FeedbackIn(BaseModel):
    query: str = Field(..., description="Текст запроса оператора/клиента")
    answer_full_text: str = Field(..., description="Полный ответ бота (full_text из /ask)")
    is_helpful: bool = Field(..., description="True — ответ полезен, False — не полезен")
    comment: Optional[str] = Field(
        None,
        description="Опциональный комментарий: что не так / что улучшить",
    )
    used_issue_keys: List[str] = Field(
        default_factory=list,
        description="Список issue_key, которые RAG использовал",
    )


class FeedbackOut(BaseModel):
    id: int
    status: str


@router.post("/", response_model=FeedbackOut)
def create_feedback(payload: FeedbackIn):
    """
    Запись одного фидбэка от оператора.
    """
    try:
        with psycopg.connect(get_db_url()) as conn, conn.cursor() as cur:
            ensure_feedback_schema(conn)
            cur.execute(
                """
                INSERT INTO feedback (query, answer_full_text, is_helpful, comment, used_issue_keys)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    payload.query,
                    payload.answer_full_text,
                    payload.is_helpful,
                    payload.comment,
                    payload.used_issue_keys or None,
                ),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return FeedbackOut(id=new_id, status="ok")

class FeedbackStats(BaseModel):
    total: int
    helpful: int
    not_helpful: int
    helpful_ratio: float  # от 0 до 1


@router.get("/stats", response_model=FeedbackStats)
def get_feedback_stats():
    """
    Простейшая онлайн-метрика качества:
    - сколько всего фидбэков
    - сколько из них полезных
    - доля полезных ответов
    """
    with psycopg.connect(get_db_url()) as conn, conn.cursor() as cur:
        ensure_feedback_schema(conn)

        cur.execute("SELECT COUNT(*) FROM feedback;")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM feedback WHERE is_helpful = TRUE;")
        helpful = cur.fetchone()[0]

    not_helpful = total - helpful
    ratio = (helpful / total) if total > 0 else 0.0

    return FeedbackStats(
        total=total,
        helpful=helpful,
        not_helpful=not_helpful,
        helpful_ratio=ratio,
    )
