# Bank RAG Agent

Сервис принимает новые заявки из Service Desk, ищет похожие кейсы в базе Qdrant и возвращает структурированные рекомендации для сотрудников сопровождения.

## Архитектура
- **Service Desk (Naumen)** отправляет новые заявки в оркестратор.
- **n8n** принимает вебхук, фильтрует запрос и вызывает `/process_ticket`.
- **RAG Core**:
  - обезличивание входного текста,
  - поиск релевантных кейсов в Qdrant,
  - генерация ответа через Ollama,
  - постобработка и форматирование.

## Запуск

```bash
export OLLAMA_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="qwen2.5:7b-instruct-q4_k_m"
export EMBEDDING_MODEL="intfloat/multilingual-e5-small"
export EMBEDDING_DEVICE="cpu"
export QDRANT_URL="http://localhost:6333"
export SERVICE_API_KEY="<optional>"

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

> Примечание: эмбеддинг-часть работает только на CPU (параметр `EMBEDDING_DEVICE` ограничен значением `cpu`). Для Ollama используйте CPU-конфигурацию на стороне сервиса.

Для гарантированного CPU-only стека зависимости PyTorch устанавливаются из CPU-репозитория (см. `requirements.txt`).

## API

### POST /process_ticket

Заголовок (опционально, если задан `SERVICE_API_KEY`):
- `X-API-Key: <ключ>`

n8n должен вызывать этот endpoint после получения тикета из Service Desk, передавая текст инцидента и `ticket_id`.

Пример запроса:

```json
{
  "ticket_id": "INC-12345",
  "text": "Описание инцидента..."
}
```

Пример успешного ответа:

```json
{
  "ticket_id": "INC-12345",
  "original_text": "Обезличенный текст...",
  "rag_response": {
    "description": ["..."],
    "causes": ["..."],
    "actions": ["..."],
    "next_steps": ["..."]
  },
  "used_context_len": 1234
}
```

Возможные ошибки:
- `401 Unauthorized` — неверный API ключ.
- `502 Bad Gateway` — LLM вернул пустой ответ.

## Зависимости
Все зависимости фиксируются в `requirements.txt` для стабильных сборок и одинакового окружения.
