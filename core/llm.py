# core/llm.py
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

import requests

from .config import OLLAMA_URL, OLLAMA_MODEL

log = logging.getLogger("rag.llm")

OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "4"))
OLLAMA_RETRY_BACKOFF_BASE_SEC = float(os.getenv("OLLAMA_RETRY_BACKOFF_BASE_SEC", "2"))

# READ timeout (ждём генерацию) + connect отдельно
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "900"))
OLLAMA_CONNECT_TIMEOUT_SEC = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_SEC", "10"))

# Ограничение длины ответа
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))

# Контекст
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# Параметры семплирования
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))


def build_prompt(context: str, question: str) -> str:
    return f"""Ты — инженер технической поддержки банка (1-я линия).
Твоя задача — дать практичные действия по новому обращению на основе истории похожих кейсов.

Правила:
- Отвечай ТОЛЬКО по-русски.
- Никакой «воды». Только конкретные действия/проверки.
- НЕ копируй контекст дословно.
- Опирайся только на факты из КОНТЕКСТА; не придумывай команды, системы и причины.
- Если данных из контекста недостаточно, прямо так и напиши и добавь, что нужно уточнить.
- Приоритет: сначала безопасные проверки (логи/доступность/права), затем изменения.
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


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json() if resp.content else {}
    except Exception:
        return {}


def _ns_to_ms(v: Any) -> float:
    try:
        # ollama отдаёт duration в наносекундах
        return float(v) / 1_000_000.0
    except Exception:
        return 0.0


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
            "num_predict": OLLAMA_NUM_PREDICT,
            "stop": ["\n\n5)", "\n\n5.", "\n\nПятый", "\n\nИтог"],
        },
    }

    last_error: Exception | None = None
    timeout = (OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_TIMEOUT_SEC)

    prompt_chars = len(prompt)
    context_chars = len(context or "")
    question_chars = len(question or "")

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

            # ---- метрики Ollama (если есть) ----
            total_ms = _ns_to_ms(data.get("total_duration"))
            load_ms = _ns_to_ms(data.get("load_duration"))
            pe_cnt = int(data.get("prompt_eval_count") or 0)
            pe_ms = _ns_to_ms(data.get("prompt_eval_duration"))
            ev_cnt = int(data.get("eval_count") or 0)
            ev_ms = _ns_to_ms(data.get("eval_duration"))

            tok_s = 0.0
            if ev_ms > 0 and ev_cnt > 0:
                tok_s = ev_cnt / (ev_ms / 1000.0)

            if answer:
                log.info(
                    "ollama_ok elapsed=%.1fs model=%s num_predict=%s "
                    "prompt_chars=%s context_chars=%s question_chars=%s "
                    "ollama_total_ms=%.1f load_ms=%.1f prompt_eval_cnt=%s prompt_eval_ms=%.1f "
                    "eval_cnt=%s eval_ms=%.1f tok_s=%.2f trace_id=%s job_id=%s",
                    dt,
                    OLLAMA_MODEL,
                    OLLAMA_NUM_PREDICT,
                    prompt_chars,
                    context_chars,
                    question_chars,
                    total_ms,
                    load_ms,
                    pe_cnt,
                    pe_ms,
                    ev_cnt,
                    ev_ms,
                    tok_s,
                    trace_id or "-",
                    job_id or "-",
                )
                return answer

            last_error = RuntimeError("LLM response is empty")
            raise last_error

        except Exception as e:
            dt = time.time() - t0
            last_error = e

            log.warning(
                "ollama_error attempt=%s/%s elapsed=%.1fs error=%s prompt_chars=%s trace_id=%s job_id=%s",
                attempt,
                OLLAMA_MAX_RETRIES,
                dt,
                e,
                prompt_chars,
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
