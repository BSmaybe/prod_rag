# api/main.py
from __future__ import annotations

import importlib.util
import os
from fastapi import FastAPI, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from core.llm import generate_answer
from core.vectordb import search_hits, search_tickets
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
):
    service_api_key = os.getenv("SERVICE_API_KEY") or os.getenv("SERVICE_DESK_API_KEY")
    if service_api_key and x_api_key != service_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    clean_text = anonymize_text(request.text)

    top_k = int(os.getenv("PROCESS_TOP_K", "5"))
    context = search_tickets(clean_text, top_k=top_k)
    if not context:
        return {"status": "no_context", "answer": "Не нашел похожих инцидентов."}

    raw_answer = generate_answer(context, clean_text)
    if not raw_answer.strip():
        raise HTTPException(status_code=502, detail="LLM response is empty")

    structured_response = to_structured(raw_answer)

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

    return {
        "ticket_id": request.ticket_id,
        "original_text": clean_text,
        "rag_response": structured_response,
        "used_context_len": len(context),
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    return {"ready": True}


app.include_router(manage_router)
app.include_router(servicedesk_router)
