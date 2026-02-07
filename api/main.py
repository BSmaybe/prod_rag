# api/main.py
from __future__ import annotations

import importlib.util
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from fastapi import FastAPI, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from api.db import db_conn, ensure_schema, set_ticket_status
from core.llm import generate_answer
from core.vectordb import search_hits
from etl.anonymize import anonymize_text
from api.utils.formatter import to_structured
from api.manage import router as manage_router
from api.routes.servicedesk import router as servicedesk_router


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


@app.on_event("startup")
def startup_init() -> None:
    try:
        with db_conn() as conn:
            ensure_schema(conn)
        log.info("tickets_inbox schema is ready")
    except Exception as e:
        log.error("failed_to_prepare_tickets_inbox error=%s", e)


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


@app.post("/ask")
def ask(
    issue_text: str = Form(...),
    context_count: int = Form(20),
    service: str | None = Form(None),
    request: Request | None = None,
):
    """
    1) Ищем контекст (RAG)
    2) Генерим ответ (LLM)
    3) Превращаем «сырой» ответ в структуру (formatter)
    4) (опц.) postprocess
    5) Возвращаем предсказуемый JSON
    """
    if not issue_text or not issue_text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "empty_issue_text", "message": "issue_text не должен быть пустым"},
        )

    # 1) Поиск контекста в Qdrant
    try:
        top_k = max(1, min(50, int(context_count)))
        results = search_hits(issue_text, top_k=top_k, service=service) or []
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "retrieval_failed", "message": f"Ошибка поиска контекста: {e}"},
        )

    if not results:
        return JSONResponse(
            status_code=404,
            content={"error": "no_documents", "message": "В Qdrant не найден релевантный контекст"},
        )

    used_issue_keys = [r[0] for r in results]
    used_snippets = [str(r[1])[:300] for r in results]
    context_chunks = [str(r[1]) for r in results]
    context = "\n\n---\n\n".join(context_chunks)

    # 2) Генерация LLM через Ollama
    try:
        raw_answer = generate_answer(context=context, question=issue_text)
        if not raw_answer.strip():
            raise RuntimeError("LLM response is empty")
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": "llm_failed", "message": f"Ошибка генерации ответа LLM: {e}"},
        )

    # 3) Форматирование → 4 секции
    try:
        structured = to_structured(raw_answer)
    except Exception as e:
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
            structured = postprocess(structured, query=issue_text, context=context)  # type: ignore
        except TypeError:
            try:
                structured = postprocess(structured)  # type: ignore
            except Exception:
                pass
        except Exception:
            # не ломаем ответ, если постпроцессор дал сбой
            pass

    if request is not None:
        log.info(
            "ask request_id=%s query_len=%s top_k=%s hits=%s",
            getattr(request.state, "request_id", "-"),
            len(issue_text),
            top_k,
            len(results),
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
        }
    )


@app.post("/process_ticket")
async def process_ticket(
    request: TicketRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    http_request: Request | None = None,
):
    service_api_key = os.getenv("SERVICE_API_KEY") or os.getenv("SERVICE_DESK_API_KEY")
    if service_api_key and x_api_key != service_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    clean_text = anonymize_text((request.text or "").strip())
    if not clean_text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    top_k = int(os.getenv("PROCESS_TOP_K", "5"))
    hits = search_hits(clean_text, top_k=top_k) or []
    top1_score = float(hits[0][2]) if hits else 0.0
    used_issue_keys = [h[0] for h in hits]

    if len(hits) < NO_CONTEXT_MIN_HITS or top1_score < NO_CONTEXT_SCORE_THRESHOLD:
        comment = _clarification_template()
        try:
            with db_conn() as conn:
                set_ticket_status(conn, ticket_id=request.ticket_id, status="no_context")
        except Exception as e:
            log.warning("process_ticket_status_update_failed ticket_id=%s error=%s", request.ticket_id, e)
        if http_request is not None:
            log.info(
                "process_ticket request_id=%s ticket_id=%s status=no_context hits=%s top1_score=%.4f",
                getattr(http_request.state, "request_id", "-"),
                request.ticket_id,
                len(hits),
                top1_score,
            )
        return {
            "ticket_id": request.ticket_id,
            "status": "no_context",
            "comment": comment,
            "used_issue_keys": used_issue_keys,
            "top1_score": top1_score,
        }

    context = "\n\n---\n\n".join([str(h[1]) for h in hits])
    raw_answer = generate_answer(context, clean_text)
    if not raw_answer.strip():
        try:
            with db_conn() as conn:
                set_ticket_status(conn, ticket_id=request.ticket_id, status="error")
        except Exception as e:
            log.warning("process_ticket_error_status_failed ticket_id=%s error=%s", request.ticket_id, e)
        raise HTTPException(status_code=502, detail="LLM response is empty")

    structured_response: dict[str, Any] = to_structured(raw_answer)

    if callable(postprocess):
        try:
            structured_response = postprocess(structured_response, query=clean_text, context=context)  # type: ignore
        except TypeError:
            try:
                structured_response = postprocess(structured_response)  # type: ignore
            except Exception:
                pass
        except Exception:
            pass

    comment = structured_response.get("full_text", raw_answer)
    try:
        with db_conn() as conn:
            set_ticket_status(conn, ticket_id=request.ticket_id, status="processed")
    except Exception as e:
        log.warning("process_ticket_status_update_failed ticket_id=%s error=%s", request.ticket_id, e)

    if http_request is not None:
        log.info(
            "process_ticket request_id=%s ticket_id=%s status=ok text_len=%s hits=%s top1_score=%.4f",
            getattr(http_request.state, "request_id", "-"),
            request.ticket_id,
            len(clean_text),
            len(hits),
            top1_score,
        )

    return {
        "ticket_id": request.ticket_id,
        "status": "ok",
        "comment": comment,
        "used_issue_keys": used_issue_keys,
        "top1_score": top1_score,
        "original_text": clean_text,
        "rag_response": structured_response,
        "used_context_len": len(hits),
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    return {"ready": True}


app.include_router(manage_router)
app.include_router(servicedesk_router)
