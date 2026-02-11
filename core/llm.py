# core/llm.py
from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from .config import OLLAMA_URL, OLLAMA_MODEL

log = logging.getLogger("rag.llm")

OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "4"))
OLLAMA_RETRY_BACKOFF_BASE_SEC = float(os.getenv("OLLAMA_RETRY_BACKOFF_BASE_SEC", "2"))

# ВАЖНО: это будет READ timeout (сколько ждём генерацию), а connect сделаем отдельно ниже
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "900"))
OLLAMA_CONNECT_TIMEOUT_SEC = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SEC", "10"))

# Ограничение длины ответа (сильно влияет на скорость)
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "500"))

# Контекст
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# Параметры семплирования
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))


def build_prompt(context: str, question: str) -> str:
    # Этот промпт взят из RGA-SD (llm/generator.py), он проверен боем
    return f"""Ты — инженер технической поддержки банка (1-я линия).
Твоя задача — дать практичные действия по новому обращению на основе истории похожих кейсов.

Правила:
- Отвечай ТОЛЬКО по-русски.
- Никакой «воды». Только конкретные действия/проверки.
- НЕ копируй контекст дословно.
- Строго 4 раздела (1–4). Никаких лишних заголовков/текста до/после.

КОНТЕКСТ (фрагменты прошлых инцидентов):
{context}

НОВЫЙ ИНЦИДЕНТ:
\"\"\"{question}\"\"\"

Формат ответа (строго):
1) Описание проблемы:
- ...
2) Возможные причины:
- ...
3) Рекомендуемые действия:
- ...
4) Следующие шаги/эскалация:
- ...
"""


def _safe_json(resp: requests.Response) -> dict[str, Any]:
    try:
        return resp.json() if resp.content else {}
    except Exception:
        return {}


def generate_answer(
    context: str,
    question: str,
    trace_id: str | None = None,
    job_id: str | None = None,
) -> str:
    prompt = build_prompt(context, question)

    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_ctx": OLLAMA_NUM_CTX,
            # КЛЮЧЕВОЕ: ограничиваем длину генерации, иначе будет очень долго
            "num_predict": OLLAMA_NUM_PREDICT,
            # Опционально: можно подсказать остановку
            # (если мешает — убери)
            "stop": [
                "\n\n5)",
                "\n\n5.",
                "\n\nПятый",
                "\n\nИтог",
            ],
        },
    }

    last_error: Exception | None = None

    # connect timeout короткий, read timeout длинный (ждём генерацию)
    timeout = (OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_TIMEOUT_SEC)

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            dt = time.time() - t0

            if resp.status_code >= 400:
                body = _safe_json(resp)
                log.warning(
                    "ollama_http_error status=%s elapsed=%.1fs body_keys=%s trace_id=%s job_id=%s",
                    resp.status_code,
                    dt,
                    list(body.keys())[:10],
                    trace_id or "-",
                    job_id or "-",
                )
                resp.raise_for_status()

            data = _safe_json(resp)
            answer = str(data.get("response", "")).strip()

            if answer:
                log.info(
                    "ollama_ok elapsed=%.1fs chars=%s model=%s trace_id=%s job_id=%s",
                    dt,
                    len(answer),
                    OLLAMA_MODEL,
                    trace_id or "-",
                    job_id or "-",
                )
                return answer

            last_error = RuntimeError("LLM response is empty")
            raise last_error

        except Exception as e:
            dt = time.time() - t0
            last_error = e

            # Полезный лог, чтобы видеть где именно падает
            log.warning(
                "ollama_error attempt=%s/%s elapsed=%.1fs error=%s trace_id=%s job_id=%s",
                attempt,
                OLLAMA_MAX_RETRIES,
                dt,
                e,
                trace_id or "-",
                job_id or "-",
            )

            if attempt < OLLAMA_MAX_RETRIES:
                delay_sec = OLLAMA_RETRY_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                log.warning(
                    "ollama_retry attempt=%s/%s delay_sec=%.1f trace_id=%s job_id=%s",
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    delay_sec,
                    trace_id or "-",
                    job_id or "-",
                )
                time.sleep(delay_sec)

    log.error(
        "ollama_failed attempts=%s error=%s trace_id=%s job_id=%s",
        OLLAMA_MAX_RETRIES,
        last_error,
        trace_id or "-",
        job_id or "-",
    )
    return ""
