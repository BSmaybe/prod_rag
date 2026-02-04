# llm/rephrase.py
import os
from functools import lru_cache
from llama_cpp import Llama

REPHRASE_MODEL_PATH = os.getenv(
    "LLM_REPHRASE_MODEL_PATH",
    os.getenv("LLM_MODEL_PATH", "models/qwen2.5-7b-instruct-q4_K_M.gguf"),
)

REPHRASE_N_CTX = int(os.getenv("LLM_REPHRASE_N_CTX", "2048"))
REPHRASE_MAX_TOKENS = int(os.getenv("LLM_REPHRASE_MAX_TOKENS", "400"))
REPHRASE_TEMPERATURE = float(os.getenv("LLM_REPHRASE_TEMPERATURE", "0.3"))
REPHRASE_THREADS = int(os.getenv("LLM_THREADS", "8"))


@lru_cache(maxsize=1)
def _get_llm() -> Llama:
    return Llama(
        model_path=REPHRASE_MODEL_PATH,
        n_ctx=REPHRASE_N_CTX,
        n_threads=REPHRASE_THREADS,
        verbose=False,
    )


def build_rephrase_prompt(text: str) -> str:
    return f"""Ты — инженер технической поддержки, который готовит описание инцидентов
для поиска похожих случаев.

Переформулируй описание так, чтобы:
- оно было технически точным;
- сохранились ключевые симптомы;
- убраны лишние детали;
- длина была 1–3 абзаца.

Верни ТОЛЬКО итоговый текст.

Исходное описание:
{text}
"""


def rephrase_issue(text: str, max_len: int = 600) -> str:
    if not text or len(text.strip()) < 30:
        return text.strip()

    llm = _get_llm()
    prompt = build_rephrase_prompt(text)

    try:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Ты аккуратно переформулируешь технические инциденты."},
                {"role": "user", "content": prompt},
            ],
            temperature=REPHRASE_TEMPERATURE,
            max_tokens=REPHRASE_MAX_TOKENS,
        )

        result = out["choices"][0]["message"]["content"].strip()
        if not result:
            return text.strip()

        return result[:max_len]

    except Exception:
        return text.strip()
