# RGA-SD

RAG-сервис для Service Desk с активной архитектурой **Qdrant + Ollama + n8n**.

## Текущая архитектура
- **Naumen callback endpoint:** `POST /sd/tickets` (фиксированный вход для событий).
- **API (`api/`)**:
  - `POST /ask` — ручной запрос (Qdrant + Ollama)
  - `POST /process_ticket` — оркестрационный endpoint для n8n
  - `POST /manage/reindex` — доиндексация `data/new_tickets/*.csv`
- **Qdrant** — векторная БД (активная).
- **Ollama** — генерация ответа.
- **n8n** — оркестрация: вызов `/process_ticket`, отправка комментария обратно в Naumen.

Важно: перед записью в Qdrant текст **обязательно анонимизируется** (`etl/anonymize.py::anonymize_text`).

## Что отключено в активном контуре
- Postgres/pgvector и `rag-cron` не используются.
- `/feedback` и `/feedback/stats` не подключены (ожидаемо `404`).

## Быстрый старт
1. Создайте `.env` на основе `.env.example`.
2. Убедитесь, что локальная embedding-модель доступна в `./local_model`.
3. Запустите стек:

```bash
docker compose up --build -d
```

4. Проверка:

```bash
curl http://localhost:8080/readyz
```

## Service Desk и n8n
- Naumen отправляет callback **напрямую** в:

```text
POST /sd/tickets
```

- n8n не подменяет callback endpoint; он используется как оркестратор поверх API.
- Workflow: `docs/n8n-workflows/naumen-ticket-processing.json`.

## Релизный шаг: очистка и перезаливка Qdrant
Принятая стратегия — очистить коллекцию и заполнить заново анонимизированными данными.

1. Остановить поток новых callback событий из Naumen.
2. Удалить коллекцию Qdrant:

```bash
curl -X DELETE "http://localhost:6333/collections/${COLLECTION_NAME:-bank_tickets}"
```

3. Перезапустить ingest/reindex (например, через `/manage/reindex` и SD-поток).
4. Проверить, что в `text_chunk` нет сырых PII.
5. Вернуть callback поток.

## API (активные)
- `POST /sd/tickets` — Naumen callback (контракт сохранён).
- `POST /ask` — ручной RAG-запрос.
- `POST /process_ticket` — endpoint для n8n.
- `POST /manage/reindex` — доиндексация CSV.
- `GET /healthz`, `GET /readyz`.

## Legacy код
Старые pgvector-скрипты и утилиты оставлены в репозитории для совместимости, но не участвуют в активном runtime-пути.
