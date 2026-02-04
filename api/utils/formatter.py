# api/utils/formatter.py
from __future__ import annotations
import re
from typing import Dict, List

SECTION_TITLES = {
    "desc": ["описание проблемы", "summary", "problem", "problem description"],
    "causes": ["возможные причины", "причины", "root cause", "hypotheses"],
    "actions": ["рекомендованные действия", "шаги", "действия", "fix", "mitigation"],
    "next": ["следующие шаги", "если не решено", "эскалация", "что дальше", "next steps"],
}


def _clean_item(s: str) -> str:
    """Убираем маркеры списков и лишние пробелы."""
    s = s.strip()
    # срезаем "1.", "1)", "-", "*", "•" в начале
    s = re.sub(r"^(\d+[\.\)]\s*|[-*•]\s*)", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _split_bullets(block: str) -> List[str]:
    """Разбиваем текст секции на элементы списка по строкам/маркерам."""
    if not block:
        return []

    # нормализуем переносы
    block = block.replace("\r", "")
    lines = [ln.rstrip() for ln in block.split("\n")]

    items: List[str] = []
    buf: List[str] = []

    def flush():
        if not buf:
            return
        item = _clean_item(" ".join(buf))
        if item:
            items.append(item)
        buf.clear()

    for ln in lines:
        if not ln.strip():
            # пустую строку считаем границей элемента
            flush()
            continue

        # новая буллета / новый нумерованный пункт
        if re.match(r"^\s*([-*•]|\d+[\.\)])\s+", ln):
            flush()
            buf.append(ln)
        else:
            buf.append(ln)

    flush()
    # убираем очень короткий/мусорный текст
    return [it for it in items if len(it) > 2]


def _pick_section_key_from_title(title: str) -> str | None:
    """По заголовку секции пытаемся понять, что это: описание / причины / действия / next."""
    t = title.lower()
    t = re.sub(r"[\s:]+", " ", t).strip()

    for key, variants in SECTION_TITLES.items():
        for v in variants:
            if v in t:
                if key == "desc":
                    return "description"
                if key == "causes":
                    return "causes"
                if key == "actions":
                    return "actions"
                if key == "next":
                    return "next_steps"
    return None


def _split_by_numbered_headers(text: str) -> Dict[str, str]:
    """Режем по заголовкам вида "1) ...", "2) ...".

    Возвращает dict label -> block, где label — строка "1","2","3","4".
    """
    if not text:
        return {}

    text_norm = text.replace("\r", "")
    # находим строки, начинающиеся с "1) ", "2) ", и т.д.
    pattern = re.compile(r"(?m)^(?P<label>\d+)[\.\)]\s*(?P<title>.+?)\s*$")
    matches = list(pattern.finditer(text_norm))
    if not matches:
        return {}

    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        label = m.group("label")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text_norm)
        block = text_norm[start:end].strip("\n ")
        sections[label] = block

    return sections


def to_structured(text: str) -> Dict[str, List[str] | str]:
    """Парсит сырой ответ LLM в структуру с 4 списками.

    Возвращает словарь:
    {
      "description": [...],
      "causes": [...],
      "actions": [...],
      "next_steps": [...],
      "full_text": "...",
    }
    """
    structured: Dict[str, List[str] | str] = {
        "description": [],
        "causes": [],
        "actions": [],
        "next_steps": [],
        "full_text": text or "",
    }

    if not text:
        return structured

    # 1) Пытаемся парсить по явной нумерации 1) 2) 3) 4)
    sections_by_num = _split_by_numbered_headers(text)
    if sections_by_num:
        for lab, block in sections_by_num.items():
            key: str | None = None
            if lab == "1":
                key = "description"
            elif lab == "2":
                key = "causes"
            elif lab == "3":
                key = "actions"
            elif lab == "4":
                key = "next_steps"

            if key:
                structured[key] = _split_bullets(block)  # type: ignore[assignment]

    # 2) Если какие-то секции не заполнились — fallback по заголовкам
    if not any(structured[k] for k in ["description", "causes", "actions", "next_steps"]):
        # пробуем искать pattern "Описание проблемы:", "Возможные причины:" и т.д.
        # строим regex на основе SECTION_TITLES
        title_parts = []
        for variants in SECTION_TITLES.values():
            for t in variants:
                title_parts.append(re.escape(t))
        title_regex = "|".join(title_parts)
        # разделяем по заголовкам
        pattern = re.compile(
            rf"(?i)(?P<title>{title_regex})\s*:\s*"
        )
        chunks = pattern.split(text.replace("\r", ""))
        # формат: [pre, title1, block1, title2, block2, ...]
        if len(chunks) >= 3:
            it = iter(chunks[1:])  # пропускаем pre
            for title, block in zip(it, it):
                key = _pick_section_key_from_title(title)
                if not key:
                    continue
                structured[key] = _split_bullets(block)  # type: ignore[assignment]

    # 3) Если вообще ничего не напарсили — кладём всё в описание
    if not any(structured[k] for k in ["description", "causes", "actions", "next_steps"]):
        structured["description"] = _split_bullets(text)  # type: ignore[assignment]

    # 4) Финальная чистка / ограничения по длине
    for k in ["description", "causes", "actions", "next_steps"]:
        lst = structured[k]  # type: ignore[index]
        if not isinstance(lst, list):
            continue
        structured[k] = [it[:400].rstrip() for it in lst[:8]]  # type: ignore[assignment]

    return structured
