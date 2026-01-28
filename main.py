from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# Наши модули
from core.llm import generate_answer
from core.vectordb import search_tickets
from core.config import SERVICE_API_KEY
from utils.anonymize import anonymize_text
from utils.formatter import to_structured
from utils.postprocess import postprocess

app = FastAPI(title="Bank RAG Agent")

class TicketRequest(BaseModel):
    ticket_id: str
    text: str

@app.post("/process_ticket")
async def process_ticket(
    request: TicketRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if SERVICE_API_KEY and x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    print(f"Обработка тикета: {request.ticket_id}")

    # 1. Обезличивание (код из RGA-SD)
    clean_text = anonymize_text(request.text)

    # 2. Поиск знаний (RAG через Qdrant)
    context = search_tickets(clean_text, top_k=5)

    if not context:
        return {"status": "no_context", "answer": "Не нашел похожих инцидентов."}

    # 3. Генерация ответа (Ollama)
    raw_answer = generate_answer(context, clean_text)
    if not raw_answer.strip():
        raise HTTPException(status_code=502, detail="LLM response is empty")

    # 4. Форматирование в структуру (JSON)
    structured_response = to_structured(raw_answer)

    # 5. Постобработка (чистка дублей, добавление дефолтных шагов)
    final_response = postprocess(structured_response)

    return {
        "ticket_id": request.ticket_id,
        "original_text": clean_text, # Возвращаем обезличенный!
        "rag_response": final_response,
        "used_context_len": len(context)
    }
