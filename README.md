# RGA-SD

RAG-сервис для Service Desk с архитектурой **FastAPI + Postgres + Qdrant + Ollama + n8n**.

## Текущая архитектура
- **Naumen callback endpoint:** `POST /sd/tickets`.
- **API (`api/`)**:
  - `POST /sd/tickets` — быстрый callback-приёмник (`202`), Naumen не блокируется.
  - `POST /process_ticket` — оркестрационный endpoint для n8n (`ticket_id`, `text`).
  - `POST /ask` — ручной RAG-запрос (`context_count` управляет количеством retrieved chunks).
  - `POST /manage/reindex` — доиндексация CSV (legacy/manual).
- **Postgres** — системный журнал `tickets_inbox` (все входящие события).
- **Qdrant (`kb_tickets`)** — только KB-кейсы (закрытые и прошедшие фильтр качества).
- **Ollama** — генерация ответа.
- **n8n** — оркестрация: webhook от API -> `/process_ticket` -> комментарий в Naumen.
- **Reranker** — включён лёгкий гибридный реранк (vector score + lexical overlap) перед LLM.
- **Идентификаторы тикета в интеграции:**
  - `ticket_id` — внутренний ID Naumen (`serviceCall$...`), используется как `source` при `createComment`.
  - `issue_key` — человекочитаемый номер (`RP...`), хранится для трассировки и логов.

## Правила обработки
1. На входе `/sd/tickets` текст обезличивается и сохраняется в Postgres (`tickets_inbox`).
2. API асинхронно вызывает `N8N_WEBHOOK_URL` (fire-and-forget).
3. `/process_ticket` делает поиск в Qdrant `kb_tickets`.
4. Gate no-context:
- `hits < 2` или `top1_score < 0.35` -> `status=no_context` и шаблон уточнения без LLM.
5. Если контекст достаточен -> LLM ответ (`status=ok`).
6. В Qdrant пишутся только закрытые тикеты, прошедшие фильтр качества.

Важно: в Qdrant пишутся только анонимизированные тексты.

## Что отключено
- `rag-cron` удалён из активного контура.
- Telegram-интеграции удалены.
- `/feedback` и `/feedback/stats` не подключены (ожидаемо `404`).

## Быстрый старт
1. Создать `.env` на базе `.env.example`.
2. Убедиться, что embedding-модель доступна в `./local_model`.
3. Запустить:

```bash
docker compose up --build -d
```

4. Проверить API:

```bash
curl http://localhost:8080/readyz
```

## Bulk ingest CSV format
`POST /manage/reindex` читает CSV из `NEW_TICKETS_DIR` с таким форматом:

- Минимальный (обратная совместимость): `issue_key,text`
- Расширенный: `issue_key,text,solution_text`

Если `solution_text` передан, в Qdrant индексируется объединённый текст `problem + solution`.
В payload сохраняются раздельные поля: `problem_text`, `solution_text`, `text_chunk`.

## Articles collection (product knowledge)
Для статей используется отдельная коллекция `ARTICLES_COLLECTION` (по умолчанию `kb_articles`).
Статьи читаются из папки `ARTICLES_DIR` (md/txt), чанкуются по абзацам.
Опционально включается LLM‑judge для очистки/структурирования (off-line при reindex).
Для продуктовых статей используйте `ARTICLES_JUDGE_MODE=product`.

Ручной запуск:
```
POST /manage/reindex_articles
```

### KB фильтры для bulk reindex
Для `/manage/reindex` включены быстрые фильтры качества (без LLM):
- `KB_REQUIRE_SOLUTION=1` — не индексировать тикеты без решения.
- `KB_MIN_SOLUTION_CHARS=30` — не индексировать короткие решения.
- `KB_STOPWORDS=...` — отсекает «отписки» вроде “Ок/Решено”.
- `KB_DEDUP_ENABLED=1` — семантический дедуп (score >= `KB_DEDUP_SCORE`).
Опционально включается LLM‑judge для тикетов (off-line при reindex):
- `KB_JUDGE_ENABLED=1`, `KB_JUDGE_MODE=incident`

## Qdrant reset/reload
1. Остановить входящий callback поток из Naumen.
2. Удалить коллекцию:

```bash
curl -X DELETE "http://localhost:6333/collections/${COLLECTION_NAME:-kb_tickets}"
```

3. Запустить повторную загрузку данных (закрытые тикеты с фильтром качества).
4. Проверить, что `text_chunk` анонимизирован.
5. Возобновить callback поток.

## Переменные окружения (ключевые)
- `DB_URL`, `POSTGRES_*`
- `QDRANT_URL`, `COLLECTION_NAME=kb_tickets`
- `ARTICLES_COLLECTION=kb_articles`, `ARTICLES_DIR=data/articles`
- `ARTICLES_MAX_CHUNK_CHARS=900`, `ARTICLES_TOP_K=2`, `ARTICLES_ANONYMIZE=0`
- `ARTICLES_JUDGE_ENABLED=0`, `ARTICLES_JUDGE_CACHE_DIR=data/articles_clean`
- `ARTICLES_JUDGE_MODE=product`
- `OLLAMA_URL`, `OLLAMA_MODEL`
- `ASK_DEFAULT_TOP_K=3`, `ASK_MAX_TOP_K=10`, `PROCESS_TOP_K=3`
- `LLM_MAX_CHUNK_CHARS=900`, `LLM_MAX_CONTEXT_CHARS=2800`, `LLM_MIN_TAIL_CHARS=220`
- `RERANK_ENABLED=true`, `RERANK_POOL_MULTIPLIER=3`
- `RERANK_WEIGHT_VECTOR=0.65`, `RERANK_WEIGHT_LEXICAL=0.35`, `RERANK_OVERLAP_BOOST=0.20`
- `KB_MIN_SOLUTION_CHARS=30`, `KB_REQUIRE_SOLUTION=1`
- `KB_STOPWORDS=решено,...`, `KB_DEDUP_ENABLED=1`, `KB_DEDUP_SCORE=0.92`, `KB_DEDUP_TOP_K=1`
- `KB_JUDGE_ENABLED=0`, `KB_JUDGE_CACHE_DIR=data/tickets_clean`
- `N8N_WEBHOOK_URL`, `WEBHOOK_URL`
- `NAUMEN_API_URL`
- `SERVICE_API_KEY`/`SERVICE_DESK_API_KEY`

## Настройка `context_count`
- `context_count` в `/ask` — это число retrieved chunks до лимитирования.
- Перед вызовом LLM контекст дополнительно ограничивается `LLM_MAX_*` параметрами.
- Чем больше `context_count`, тем выше latency (`prompt_eval`) и дольше ответ.
- Рекомендованный быстрый профиль: `context_count=3`, `PROCESS_TOP_K=3`, `LLM_MAX_CONTEXT_CHARS=2800`.

## Логи
HTTP middleware логирует `request_ts` и `duration_ms` для каждого запроса.
