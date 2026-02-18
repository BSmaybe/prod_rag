# llm/generator.py
import os
from functools import lru_cache
from llama_cpp import Llama

# Путь к GGUF модели (в контейнере)
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    "/app/models/qwen2.5-7b-instruct-q4_K_M.gguf",
)

LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "768"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_THREADS = int(os.getenv("LLM_THREADS", "8"))


@lru_cache(maxsize=1)
def _get_llm() -> Llama:
    return Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_N_CTX,
        n_threads=LLM_THREADS,
        verbose=False,
    )


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

КОНТЕКСТ (фрагменты прошлых инцидентов, есть шум):
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


def generate(context: str, question: str) -> str:
    llm = _get_llm()
    prompt = build_prompt(context, question)

    out = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты опытный инженер технической поддержки банка. "
                    "Давай только проверяемые и безопасные рекомендации, "
                    "строго по контексту инцидентов."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        max_tokens=LLM_MAX_TOKENS,
        stop=["\n\n\n", "КОНТЕКСТ:", "НОВЫЙ ИНЦИДЕНТ:"],  # чтобы не повторял промпт
    )

    return out["choices"][0]["message"]["content"].strip()
