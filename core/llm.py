import logging
import os
import time

import requests

from .config import OLLAMA_URL, OLLAMA_MODEL

log = logging.getLogger("rag.llm")
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "4"))
OLLAMA_RETRY_BACKOFF_BASE_SEC = float(os.getenv("OLLAMA_RETRY_BACKOFF_BASE_SEC", "2"))
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))


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


def generate_answer(context: str, question: str) -> str:
    prompt = build_prompt(context, question)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }

    last_error: Exception | None = None

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
            response.raise_for_status()
            answer = str(response.json().get("response", "")).strip()
            if answer:
                return answer
            last_error = RuntimeError("LLM response is empty")
            raise last_error
        except Exception as e:
            last_error = e
            if attempt < OLLAMA_MAX_RETRIES:
                delay_sec = OLLAMA_RETRY_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                log.warning(
                    "ollama_retry attempt=%s/%s delay_sec=%.1f error=%s",
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    delay_sec,
                    e,
                )
                time.sleep(delay_sec)

    log.error("ollama_failed attempts=%s error=%s", OLLAMA_MAX_RETRIES, last_error)
    return ""
