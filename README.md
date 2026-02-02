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

## Docker Compose

Запуск сервисов:

```bash
docker compose up -d
```

После запуска n8n будет доступен на `http://localhost:5678` (значения можно менять через `N8N_HOST` и `N8N_PORT`).

Пример переменных окружения для `app`:

```bash
export OLLAMA_URL="http://ollama:11434/api/generate"
export QDRANT_URL="http://qdrant:6333"
export EMBEDDING_MODEL="intfloat/multilingual-e5-small"
export EMBEDDING_DEVICE="cpu"
export SERVICE_API_KEY=""
```

Пример переменных окружения для `n8n`:

```bash
export N8N_HOST="localhost"
export N8N_PORT="5678"
export N8N_BASIC_AUTH_USER="admin"
export N8N_BASIC_AUTH_PASSWORD="change-me"
export WEBHOOK_URL="http://localhost:5678"
export NAUMEN_API_URL="https://naumen.example/api/tickets"
export TELEGRAM_CHAT_ID="123456789"
```

> В Docker Compose n8n обращается к сервису `app` по адресу `http://app:8000/process_ticket`. Если n8n запускается вне Docker-сети, поменяйте URL в workflow на `http://host.docker.internal:8000/process_ticket`.

### Импорт workflow в n8n

1. Откройте n8n: `http://localhost:5678`.
2. В меню **Workflows → Import from File** выберите файл `docs/n8n-workflows/naumen-ticket-processing.json`.
3. В основном workflow обновите:
   - URL HTTP Request на `/process_ticket` (если нужно переопределить `app` ↔ `host.docker.internal`);
   - `X-API-Key`, если используется `SERVICE_API_KEY`.
4. Создайте креденшлы:
   - **HTTP Header Auth** для Naumen (используется узлом **Send comment to Naumen**).
   - **Telegram** для error workflow.
5. Включите error workflow (**Naumen Ticket Error Handler**) и назначьте его как обработчик ошибок для основного workflow в настройках.

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
