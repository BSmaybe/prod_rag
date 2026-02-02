import requests
import json
from .config import OLLAMA_URL, OLLAMA_MODEL

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
            "temperature": 0.2, # Низкая температура для точности (как в RGA-SD)
            "num_ctx": 4096
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Ошибка Ollama: {e}")
        return ""