# api/main.py
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from api.db import (
    claim_outbox_batch,
    claim_ticket_jobs_batch,
    db_conn,
    enqueue_outbox_event,
    enqueue_ticket_job,
    ensure_schema,
    get_ticket_job,
    mark_outbox_dead,
    mark_outbox_retry,
    mark_outbox_sent,
    mark_ticket_job_done,
    mark_ticket_job_error,
    set_ticket_status,
)
from api.manage import router as manage_router
from api.routes.servicedesk import router as servicedesk_router
from api.utils.formatter import to_structured
from core.llm import generate_answer
from core.vectordb import search_hits
from etl.anonymize import anonymize_text


def _load_postprocess_func():
    """Load optional formatter postprocessor if the module is present."""
    spec = importlib.util.find_spec("api.utils.postprocess")
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "postprocess", None)


postprocess = _load_postprocess_func()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("rag.api")

APP_TITLE = os.getenv("APP_TITLE", "RAG Agent API")
app = FastAPI(title=APP_TITLE)

# CORS (на время разработки)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NO_CONTEXT_SCORE_THRESHOLD = float(os.getenv("NO_CONTEXT_SCORE_THRESHOLD", "0.35"))
NO_CONTEXT_MIN_HITS = int(os.getenv("NO_CONTEXT_MIN_HITS", "2"))

OUTBOX_POLL_SEC = float(os.getenv("OUTBOX_POLL_SEC", "2"))
OUTBOX_BATCH_SIZE = int(os.getenv("OUTBOX_BATCH_SIZE", "50"))
OUTBOX_MAX_ATTEMPTS = int(os.getenv("OUTBOX_MAX_ATTEMPTS", "8"))
OUTBOX_BACKOFF_BASE_SEC = int(os.getenv("OUTBOX_BACKOFF_BASE_SEC", "5"))
JOB_POLL_SEC = float(os.getenv("JOB_POLL_SEC", "1"))
JOB_BATCH_SIZE = int(os.getenv("JOB_BATCH_SIZE", "20"))
PROCESS_TICKET_ASYNC = (os.getenv("PROCESS_TICKET_ASYNC", "true").lower() in {"1", "true", "yes", "on"})

N8N_WEBHOOK_URL = (os.getenv("N8N_WEBHOOK_URL") or "").strip()

outbox_worker_task: asyncio.Task[None] | None = None
ticket_job_worker_task: asyncio.Task[None] | None = None


@contextmanager
def _stage(timings: dict[str, float], key: str):
    """Context manager to record stage duration into timings dict in milliseconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = (time.perf_counter() - t0) * 1000.0


def _next_retry_at(attempts: int) -> datetime:
    delay_sec = OUTBOX_BACKOFF_BASE_SEC * (2 ** max(attempts, 0))
    return datetime.now(timezone.utc) + timedelta(seconds=delay_sec)


async def _send_outbox_payload(payload: dict[str, Any]) -> None:
    if not N8N_WEBHOOK_URL:
        raise RuntimeError("N8N_WEBHOOK_URL is empty")
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
        resp = await client.post(N8N_WEBHOOK_URL, json=payload)
        resp.raise_for_status()


def _claim_batch_sync(limit: int) -> list[dict[str, Any]]:
    with db_conn() as conn:
        return claim_outbox_batch(conn, limit=limit)


def _mark_sent_sync(outbox_id: int) -> None:
    with db_conn() as conn:
        mark_outbox_sent(conn, outbox_id=outbox_id)


def _mark_retry_sync(
    outbox_id: int,
    attempts: int,
    last_error: str,
    next_retry_at: datetime,
) -> None:
    with db_conn() as conn:
        mark_outbox_retry(
            conn,
            outbox_id=outbox_id,
            attempts=attempts,
            last_error=last_error,
            next_retry_at=next_retry_at,
        )


def _mark_dead_sync(outbox_id: int, last_error: str) -> None:
    with db_conn() as conn:
        mark_outbox_dead(conn, outbox_id=outbox_id, last_error=last_error)


def _enqueue_ticket_job_sync(ticket_id: str, text_anonymized: str) -> str:
    with db_conn() as conn:
        return enqueue_ticket_job(conn, ticket_id=ticket_id, text_anonymized=text_anonymized)


def _claim_ticket_jobs_batch_sync(limit: int) -> list[dict[str, Any]]:
    with db_conn() as conn:
        return claim_ticket_jobs_batch(conn, limit=limit)


def _mark_ticket_job_done_and_enqueue_result_sync(
    job_id: str,
    ticket_id: str,
    result_payload: dict[str, Any],
) -> int:
    with db_conn() as conn:
        mark_ticket_job_done(conn, job_id=job_id, result_payload=result_payload, autocommit=False)
        outbox_id = enqueue_outbox_event(
            conn,
            ticket_id=ticket_id,
            payload={
                "event_type": "ticket_result",
                "job_id": job_id,
                "ticket_id": ticket_id,
                "status": result_payload.get("status", "error"),
                "comment": result_payload.get("comment", ""),
                "used_issue_keys": result_payload.get("used_issue_keys", []),
                "top1_score": float(result_payload.get("top1_score", 0.0) or 0.0),
                "used_chunks": result_payload.get("used_chunks", []),
            },
            autocommit=False,
        )
        conn.commit()
        return outbox_id


def _mark_ticket_job_error_sync(job_id: str, error: str) -> None:
    with db_conn() as conn:
        mark_ticket_job_error(conn, job_id=job_id, error=error)


def _get_ticket_job_sync(job_id: str) -> dict[str, Any] | None:
    with db_conn() as conn:
        return get_ticket_job(conn, job_id)


async def _outbox_worker_loop() -> None:
    log.info(
        "outbox_worker_started poll_sec=%s batch_size=%s max_attempts=%s",
        OUTBOX_POLL_SEC,
        OUTBOX_BATCH_SIZE,
        OUTBOX_MAX_ATTEMPTS,
    )
    while True:
        try:
            batch = await asyncio.to_thread(_claim_batch_sync, OUTBOX_BATCH_SIZE)
            if not batch:
                await asyncio.sleep(OUTBOX_POLL_SEC)
                continue

            for item in batch:
                outbox_id = int(item.get("id") or 0)
                ticket_id = str(item.get("ticket_id") or "")
                payload = item.get("payload") or {}
                attempts = int(item.get("attempts") or 0)
                created_at = item.get("created_at")
                queue_wait_ms = -1.0
                if isinstance(created_at, datetime):
                    queue_wait_ms = (
                        datetime.now(timezone.utc) - created_at.astimezone(timezone.utc)
                    ).total_seconds() * 1000.0
                try:
                    if not isinstance(payload, dict):
                        raise RuntimeError("outbox payload must be JSON object")

                    t_deliver = time.perf_counter()
                    await _send_outbox_payload(payload)
                    deliver_ms = (time.perf_counter() - t_deliver) * 1000.0
                    await asyncio.to_thread(_mark_sent_sync, outbox_id)
                    log.info(
                        "outbox_sent outbox_id=%s ticket_id=%s attempts=%s queue_wait_ms=%.1f deliver_ms=%.1f",
                        outbox_id,
                        ticket_id,
                        attempts,
                        queue_wait_ms,
                        deliver_ms,
                    )
                except Exception as e:
                    next_attempts = attempts + 1
                    if next_attempts >= OUTBOX_MAX_ATTEMPTS:
                        await asyncio.to_thread(_mark_dead_sync, outbox_id, str(e))
                        log.error(
                            "outbox_dead outbox_id=%s ticket_id=%s attempts=%s error=%s",
                            outbox_id,
                            ticket_id,
                            next_attempts,
                            e,
                        )
                    else:
                        next_retry = _next_retry_at(next_attempts)
                        await asyncio.to_thread(
                            _mark_retry_sync,
                            outbox_id,
                            next_attempts,
                            str(e),
                            next_retry,
                        )
                        log.warning(
                            "outbox_retry outbox_id=%s ticket_id=%s attempts=%s next_retry_at=%s error=%s",
                            outbox_id,
                            ticket_id,
                            next_attempts,
                            next_retry.isoformat(),
                            e,
                        )
        except asyncio.CancelledError:
            log.info("outbox_worker_stopped")
            raise
        except Exception as e:
            log.exception("outbox_worker_error error=%s", e)
            await asyncio.sleep(OUTBOX_POLL_SEC)


def _clarification_template() -> str:
    return (
        "Недостаточно данных для точной рекомендации. Пожалуйста, уточните:\n"
        "1) Сервис/компонент (микросервис/модуль).\n"
        "2) Точный текст ошибки/код.\n"
        "3) Время инцидента (±5 минут) и среда (prod/test).\n"
        "4) Шаги воспроизведения.\n"
        "5) Логи/скрин и correlation-id/trace-id.\n"
        "6) Что уже проверяли до обращения."
    )


def _run_ticket_reasoning(
    *,
    ticket_id: str,
    clean_text: str,
    trace_id: str = "-",
    job_id: str | None = None,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    retrieval_ms = 0.0
    llm_ms = 0.0
    format_ms = 0.0

    try:
        top_k = int(os.getenv("PROCESS_TOP_K", "5"))

        # ---- retrieval
        t_retrieval = time.perf_counter()
        hits = search_hits(clean_text, top_k=top_k) or []
        used_issue_keys, used_snippets, context_chunks, scores = _hits_to_context(hits)
        top1_score = float(scores[0]) if scores else 0.0
        context_chunks_nonempty = [c for c in context_chunks if c and c.strip()]
        if not context_chunks_nonempty:
            top1_score = 0.0
        retrieval_ms = (time.perf_counter() - t_retrieval) * 1000.0

        if len(hits) < NO_CONTEXT_MIN_HITS or top1_score < NO_CONTEXT_SCORE_THRESHOLD:
            comment = _clarification_template()
            try:
                with db_conn() as conn:
                    set_ticket_status(conn, ticket_id=ticket_id, status="no_context")
            except Exception as e:
                log.warning(
                    "process_ticket_status_update_failed ticket_id=%s error=%s",
                    ticket_id,
                    e,
                )

            total_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "ticket_id": ticket_id,
                "status": "no_context",
                "comment": comment,
                "used_issue_keys": used_issue_keys,
                "top1_score": top1_score,
                "used_chunks": used_snippets,
                "used_context_len": len(hits),
                "original_text": clean_text,
                "retrieval_ms": retrieval_ms,
                "llm_ms": llm_ms,
                "format_ms": format_ms,
                "total_ms": total_ms,
            }

        context = "\n\n---\n\n".join(context_chunks_nonempty)

        # ---- llm
        t_llm = time.perf_counter()
        raw_answer = generate_answer(
            context=context,
            question=clean_text,
            trace_id=trace_id,
            job_id=job_id,
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0

        if not raw_answer.strip():
            raise RuntimeError("LLM response is empty")

        # ---- formatter/postprocess
        t_format = time.perf_counter()
        structured_response: dict[str, Any] = to_structured(raw_answer)

        if callable(postprocess):
            try:
                structured_response = postprocess(
                    structured_response, query=clean_text, context=context
                )  # type: ignore
            except TypeError:
                try:
                    structured_response = postprocess(structured_response)  # type: ignore
                except Exception:
                    pass
            except Exception:
                pass
        format_ms = (time.perf_counter() - t_format) * 1000.0

        comment = structured_response.get("full_text", raw_answer)

        try:
            with db_conn() as conn:
                set_ticket_status(conn, ticket_id=ticket_id, status="processed")
        except Exception as e:
            log.warning(
                "process_ticket_status_update_failed ticket_id=%s error=%s",
                ticket_id,
                e,
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "ticket_id": ticket_id,
            "status": "ok",
            "comment": comment,
            "used_issue_keys": used_issue_keys,
            "top1_score": top1_score,
            "original_text": clean_text,
            "rag_response": structured_response,
            "used_context_len": len(hits),
            "used_chunks": used_snippets,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "format_ms": format_ms,
            "total_ms": total_ms,
        }
    except Exception:
        try:
            with db_conn() as conn:
                set_ticket_status(conn, ticket_id=ticket_id, status="error")
        except Exception as e:
            log.warning(
                "process_ticket_error_status_failed ticket_id=%s error=%s",
                ticket_id,
                e,
            )
        raise


async def _ticket_job_worker_loop() -> None:
    log.info(
        "ticket_job_worker_started poll_sec=%s batch_size=%s",
        JOB_POLL_SEC,
        JOB_BATCH_SIZE,
    )
    while True:
        try:
            batch = await asyncio.to_thread(_claim_ticket_jobs_batch_sync, JOB_BATCH_SIZE)
            if not batch:
                await asyncio.sleep(JOB_POLL_SEC)
                continue

            for item in batch:
                job_id = str(item.get("job_id") or "")
                ticket_id = str(item.get("ticket_id") or "")
                clean_text = str(item.get("text_anonymized") or "")
                created_at = item.get("created_at")
                started_at = item.get("started_at")
                queue_wait_ms = -1.0
                if isinstance(created_at, datetime) and isinstance(started_at, datetime):
                    queue_wait_ms = (
                        started_at.astimezone(timezone.utc)
                        - created_at.astimezone(timezone.utc)
                    ).total_seconds() * 1000.0

                t_total = time.perf_counter()
                try:
                    result = await asyncio.to_thread(
                        _run_ticket_reasoning,
                        ticket_id=ticket_id,
                        clean_text=clean_text,
                        trace_id=f"job:{job_id}",
                        job_id=job_id,
                    )
                    outbox_id = await asyncio.to_thread(
                        _mark_ticket_job_done_and_enqueue_result_sync,
                        job_id,
                        ticket_id,
                        result,
                    )
                    total_ms = (time.perf_counter() - t_total) * 1000.0
                    log.info(
                        "process_job_timing job_id=%s ticket_id=%s status=%s queue_wait_ms=%.1f "
                        "retrieval_ms=%.1f llm_ms=%.1f format_ms=%.1f total_ms=%.1f outbox_id=%s",
                        job_id,
                        ticket_id,
                        result.get("status", "error"),
                        queue_wait_ms,
                        float(result.get("retrieval_ms", 0.0) or 0.0),
                        float(result.get("llm_ms", 0.0) or 0.0),
                        float(result.get("format_ms", 0.0) or 0.0),
                        total_ms,
                        outbox_id,
                    )
                except Exception as e:
                    total_ms = (time.perf_counter() - t_total) * 1000.0
                    await asyncio.to_thread(_mark_ticket_job_error_sync, job_id, str(e))
                    log.exception(
                        "process_job_failed job_id=%s ticket_id=%s queue_wait_ms=%.1f total_ms=%.1f error=%s",
                        job_id,
                        ticket_id,
                        queue_wait_ms,
                        total_ms,
                        e,
                    )
        except asyncio.CancelledError:
            log.info("ticket_job_worker_stopped")
            raise
        except Exception as e:
            log.exception("ticket_job_worker_error error=%s", e)
            await asyncio.sleep(JOB_POLL_SEC)


@app.on_event("startup")
def startup_init() -> None:
    try:
        with db_conn() as conn:
            ensure_schema(conn)
        log.info("tickets_inbox schema is ready")
    except Exception as e:
        log.error("failed_to_prepare_tickets_inbox error=%s", e)


@app.on_event("startup")
async def startup_outbox_worker() -> None:
    global outbox_worker_task
    if outbox_worker_task is None or outbox_worker_task.done():
        outbox_worker_task = asyncio.create_task(
            _outbox_worker_loop(), name="n8n-outbox-worker"
        )


@app.on_event("startup")
async def startup_ticket_job_worker() -> None:
    global ticket_job_worker_task
    if ticket_job_worker_task is None or ticket_job_worker_task.done():
        ticket_job_worker_task = asyncio.create_task(
            _ticket_job_worker_loop(), name="ticket-job-worker"
        )


@app.on_event("shutdown")
async def shutdown_outbox_worker() -> None:
    global outbox_worker_task, ticket_job_worker_task
    if outbox_worker_task is not None and not outbox_worker_task.done():
        outbox_worker_task.cancel()
        try:
            await outbox_worker_task
        except asyncio.CancelledError:
            pass
    outbox_worker_task = None
    if ticket_job_worker_task is not None and not ticket_job_worker_task.done():
        ticket_job_worker_task.cancel()
        try:
            await ticket_job_worker_task
        except asyncio.CancelledError:
            pass
    ticket_job_worker_task = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    request_ts = datetime.now(timezone.utc).isoformat()
    request.state.request_ts = request_ts
    start = time.perf_counter()
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        log.exception(
            "request_failed request_id=%s request_ts=%s method=%s path=%s",
            request_id,
            request_ts,
            request.method,
            request.url.path,
        )
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        client_ip = request.client.host if request.client else "-"
        content_length = request.headers.get("content-length", "-")
        log.info(
            "request_id=%s request_ts=%s method=%s path=%s status=%s duration_ms=%.1f client=%s content_length=%s",
            request_id,
            request_ts,
            request.method,
            request.url.path,
            status_code,
            duration_ms,
            client_ip,
            content_length,
        )
        if response is not None:
            response.headers["X-Request-ID"] = request_id


FORM_HTML = """<form method='post' action='/ask' style="font-family:ui-sans-serif">
  <label>Текст запроса:</label><br>
  <textarea name='issue_text' rows=6 cols=80 placeholder='Опишите проблему…'></textarea><br><br>
  <label>Сколько контекстов (top_k):</label>
  <input type='number' name='context_count' value='20' min='1' max='50' />
  <button type='submit'>Ask</button>
</form>"""


class TicketRequest(BaseModel):
    ticket_id: str
    text: str


@app.get("/", response_class=HTMLResponse)
def form_root():
    return FORM_HTML


# важно: браузер часто открывает /ask как GET — отдадим форму, а не 500
@app.get("/ask", response_class=HTMLResponse)
def form_alias():
    return FORM_HTML


def _extract_text_from_payload(payload: Any) -> str:
    """
    Унификация: текст мог лежать в разных ключах.
    """
    if not isinstance(payload, dict):
        return ""
    text = (
        payload.get("text_chunk")
        or payload.get("problem_text")
        or payload.get("text")
        or payload.get("issue_text")
        or payload.get("description")
        or payload.get("content")
        or payload.get("body")
        or ""
    )
    return str(text)


def _hit_to_row(h: Any) -> tuple[Any, str, float]:
    """
    Приводим один hit к (key, text, score) с максимальной совместимостью.

    Поддерживаем:
      1) Qdrant-like dict: {"id"/"point_id":..., "score":..., "payload":{...}}
      2) Наш кастом dict: {"issue_key":..., "text":..., "score":...}
      3) Legacy tuple/list: (key, text, score) или (key, text)
    """
    # dict формат
    if isinstance(h, dict):
        # --- Qdrant-like
        if ("payload" in h) or ("score" in h and ("id" in h or "point_id" in h)):
            key = h.get("id") or h.get("point_id") or h.get("issue_key")
            score = float(h.get("score", 0.0) or 0.0)
            payload = h.get("payload") or {}
            text = _extract_text_from_payload(payload)
            if not text:
                text = str(h.get("text") or h.get("issue_text") or h.get("content") or "")
            return key, text, score

        # --- кастомный dict
        key = h.get("issue_key") or h.get("id") or h.get("key")
        text = str(h.get("text") or h.get("issue_text") or h.get("content") or "")
        score = float(h.get("score", 0.0) or 0.0)
        return key, text, score

    # legacy tuple/list
    if isinstance(h, (tuple, list)):
        key = h[0] if len(h) > 0 else None
        text = str(h[1]) if len(h) > 1 and h[1] is not None else ""
        score = float(h[2]) if len(h) > 2 and h[2] is not None else 0.0
        return key, text, score

    return None, "", 0.0


def _hits_to_context(hits: list[Any]) -> tuple[list[Any], list[str], list[str], list[float]]:
    """
    На вход: list[Any] (любой формат search_hits)
    На выход: used_ids, snippets, chunks, scores
    """
    used_ids: list[Any] = []
    snippets: list[str] = []
    chunks: list[str] = []
    scores: list[float] = []

    for h in hits or []:
        key, text, score = _hit_to_row(h)
        used_ids.append(key)
        scores.append(float(score))
        chunks.append(text or "")
        snippets.append((text or "")[:300])

    return used_ids, snippets, chunks, scores


@app.post("/ask")
def ask(
    request: Request,
    issue_text: str = Form(...),
    context_count: int = Form(20),
    service: str | None = Form(None),  # оставили в API, но не передаём в search_hits
):
    """
    1) Ищем контекст (RAG)
    2) Генерим ответ (LLM)
    3) Превращаем «сырой» ответ в структуру (formatter)
    4) (опц.) postprocess
    5) Возвращаем предсказуемый JSON

    + Разметка (timings) для понимания "где затык"
    """
    issue_text = (issue_text or "").strip()
    if not issue_text:
        return JSONResponse(
            status_code=400,
            content={"error": "empty_issue_text", "message": "issue_text не должен быть пустым"},
        )

    timings: dict[str, float] = {}
    total_t0 = time.perf_counter()

    # 1) Поиск контекста в Qdrant
    try:
        top_k = max(1, min(50, int(context_count)))
        with _stage(timings, "retrieval_ms"):
            results = search_hits(issue_text, top_k=top_k) or []
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "retrieval_failed", "message": f"Ошибка поиска контекста: {e}"},
        )

    if not results:
        timings["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        log.info(
            "ask_timing request_id=%s query_len=%s top_k=%s hits=0 "
            "retrieval_ms=%.1f total_ms=%.1f",
            getattr(request.state, "request_id", "-"),
            len(issue_text),
            top_k,
            float(timings.get("retrieval_ms", 0.0) or 0.0),
            float(timings.get("total_ms", 0.0) or 0.0),
        )
        return JSONResponse(
            status_code=404,
            content={"error": "no_documents", "message": "В Qdrant не найден релевантный контекст"},
        )

    used_issue_keys, used_snippets, context_chunks, scores = _hits_to_context(results)

    context_chunks_nonempty = [c for c in context_chunks if c and c.strip()]
    if not context_chunks_nonempty:
        timings["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        log.info(
            "ask_timing request_id=%s query_len=%s top_k=%s hits=%s "
            "retrieval_ms=%.1f stage=context_parse_failed total_ms=%.1f",
            getattr(request.state, "request_id", "-"),
            len(issue_text),
            top_k,
            len(results),
            float(timings.get("retrieval_ms", 0.0) or 0.0),
            float(timings.get("total_ms", 0.0) or 0.0),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "context_parse_failed",
                "message": "Контекст найден, но не удалось извлечь текст из hits (payload). Проверь ключи payload при upsert.",
            },
        )

    context = "\n\n---\n\n".join(context_chunks_nonempty)

    # 2) Генерация LLM через Ollama
    try:
        with _stage(timings, "llm_ms"):
            raw_answer = generate_answer(
                context=context,
                question=issue_text,
                trace_id=getattr(request.state, "request_id", "-"),
            )
        if not raw_answer.strip():
            raise RuntimeError("LLM response is empty")
    except Exception as e:
        timings["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        log.info(
            "ask_timing request_id=%s query_len=%s top_k=%s hits=%s top1_score=%.4f "
            "retrieval_ms=%.1f llm_ms=%.1f total_ms=%.1f llm_failed=%s",
            getattr(request.state, "request_id", "-"),
            len(issue_text),
            top_k,
            len(results),
            float(scores[0]) if scores else 0.0,
            float(timings.get("retrieval_ms", 0.0) or 0.0),
            float(timings.get("llm_ms", 0.0) or 0.0),
            float(timings.get("total_ms", 0.0) or 0.0),
            str(e),
        )
        return JSONResponse(
            status_code=502,
            content={"error": "llm_failed", "message": f"Ошибка генерации ответа LLM: {e}"},
        )

    # 3) Форматирование → 4 секции
    try:
        with _stage(timings, "format_ms"):
            structured = to_structured(raw_answer)
    except Exception as e:
        timings["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        log.info(
            "ask_timing request_id=%s query_len=%s top_k=%s hits=%s "
            "retrieval_ms=%.1f llm_ms=%.1f format_ms=%.1f total_ms=%.1f format_failed=%s",
            getattr(request.state, "request_id", "-"),
            len(issue_text),
            top_k,
            len(results),
            float(timings.get("retrieval_ms", 0.0) or 0.0),
            float(timings.get("llm_ms", 0.0) or 0.0),
            float(timings.get("format_ms", 0.0) or 0.0),
            float(timings.get("total_ms", 0.0) or 0.0),
            str(e),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "format_failed",
                "message": f"Ошибка форматирования: {e}",
                "raw_answer": raw_answer[:1000],
            },
        )

    # 4) (опционально) пост-обработка
    if callable(postprocess):
        try:
            t0 = time.perf_counter()
            structured = postprocess(structured, query=issue_text, context=context)  # type: ignore
            timings["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        except TypeError:
            try:
                t0 = time.perf_counter()
                structured = postprocess(structured)  # type: ignore
                timings["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0
            except Exception:
                pass
        except Exception:
            pass

    timings["total_ms"] = (time.perf_counter() - total_t0) * 1000.0

    # лог метрик — ключевой
    log.info(
        "ask_timing request_id=%s query_len=%s top_k=%s hits=%s top1_score=%.4f "
        "ctx_chunks=%s ctx_chars=%s "
        "retrieval_ms=%.1f llm_ms=%.1f format_ms=%.1f postprocess_ms=%.1f total_ms=%.1f",
        getattr(request.state, "request_id", "-"),
        len(issue_text),
        top_k,
        len(results),
        float(scores[0]) if scores else 0.0,
        len(context_chunks_nonempty),
        len(context),
        float(timings.get("retrieval_ms", 0.0) or 0.0),
        float(timings.get("llm_ms", 0.0) or 0.0),
        float(timings.get("format_ms", 0.0) or 0.0),
        float(timings.get("postprocess_ms", 0.0) or 0.0),
        float(timings.get("total_ms", 0.0) or 0.0),
    )

    # 5) Ответ
    return JSONResponse(
        {
            "query": issue_text,
            "context_count": len(results),
            "description": structured.get("description", []),
            "causes": structured.get("causes", []),
            "actions": structured.get("actions", []),
            "next_steps": structured.get("next_steps", []),
            "full_text": structured.get("full_text", raw_answer),
            "used_chunks": used_snippets,
            "used_issue_keys": used_issue_keys,
            "top1_score": float(scores[0]) if scores else 0.0,
        }
    )


@app.post("/process_ticket")
async def process_ticket(
    request: TicketRequest,
    http_request: Request,
):
    clean_text = anonymize_text((request.text or "").strip())
    if not clean_text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    request_id = getattr(http_request.state, "request_id", "-")

    if PROCESS_TICKET_ASYNC:
        job_id = await asyncio.to_thread(
            _enqueue_ticket_job_sync,
            request.ticket_id,
            clean_text,
        )
        log.info(
            "process_ticket_accepted request_id=%s ticket_id=%s job_id=%s async=%s",
            request_id,
            request.ticket_id,
            job_id,
            PROCESS_TICKET_ASYNC,
        )
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "ticket_id": request.ticket_id,
                "job_id": job_id,
            },
        )

    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            _run_ticket_reasoning,
            ticket_id=request.ticket_id,
            clean_text=clean_text,
            trace_id=request_id,
        )
    except Exception as e:
        log.exception(
            "process_ticket_failed request_id=%s ticket_id=%s error=%s",
            request_id,
            request.ticket_id,
            e,
        )
        raise HTTPException(status_code=502, detail=str(e))

    total_ms = (time.perf_counter() - started) * 1000.0
    log.info(
        "process_ticket_timing request_id=%s ticket_id=%s status=%s retrieval_ms=%.1f llm_ms=%.1f format_ms=%.1f total_ms=%.1f",
        request_id,
        request.ticket_id,
        result.get("status", "error"),
        float(result.get("retrieval_ms", 0.0) or 0.0),
        float(result.get("llm_ms", 0.0) or 0.0),
        float(result.get("format_ms", 0.0) or 0.0),
        total_ms,
    )
    return result


@app.get("/process_ticket/jobs/{job_id}")
async def get_process_ticket_job(job_id: str):
    job = await asyncio.to_thread(_get_ticket_job_sync, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    return {"ready": True}


app.include_router(manage_router)
app.include_router(servicedesk_router)
