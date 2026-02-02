#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import json
from pathlib import Path
from typing import Any

# Алфавиты (латиница + кириллица с казахскими буквами)
UP = "A-ZА-ЯЁӘІҢҒҮҰҚӨҺ"
LW = "a-zа-яёәіңғүұқөһ"

# 1) Поля с префиксами: маскируем правую часть до конца токена/строки
FIELD_PATTERNS = [
    # ФИО / Имя / Фамилия
    (re.compile(r"(ФИО\s*клиента\s*:?[:]?\s*)([^,\n\r]+)", re.IGNORECASE), r"\1[NAME]"),
    (re.compile(r"(Имя\s*клиента\s*:?[:]?\s*)([^,\n\r]+)", re.IGNORECASE), r"\1[NAME]"),
    (re.compile(r"(Фамилия\s*:?[:]?\s*)([^,\n\r]+)", re.IGNORECASE), r"\1[NAME]"),

    # ИИН/БИН (в т.ч. «БИН/ИИН», «ИИН:», «ИИН клиента::»)
    (re.compile(r"(ИИН\s*клиента\s*:?[:]?\s*)[0-9 \-]{6,}", re.IGNORECASE), r"\1[IIN/BIN]"),
    (re.compile(r"(БИН\s*/\s*ИИН\s*:?[:]?\s*)[0-9 \-]{6,}", re.IGNORECASE), r"\1[IIN/BIN]"),
    (re.compile(r"(ИИН\s*[#:/ ]\s*)[0-9 \-]{6,}", re.IGNORECASE), r"\1[IIN/BIN]"),

    # Телефон с меткой
    (re.compile(r"(Номер\s*телефона\s*клиента\s*:?[:]?\s*)\+?\d[\d \-\(\)]{6,}", re.IGNORECASE), r"\1[PHONE]"),

    # Вложения (часто содержат имена) — затираем список целиком
    (re.compile(r"(Вложите\s*скрин\s*ошибки\s*:?[:]?\s*)(.+)", re.IGNORECASE), r"\1[ATTACHMENTS]"),
]

# 2) Общие «сырые» шаблоны: добиваем остатки
COMMON_PATTERNS = [
    (re.compile(r"\b\d{12}\b"), "[IIN/BIN]"),                     # ИИН/БИН: 12 подряд
    (re.compile(r"\b\d{16,19}\b"), "[CARD]"),                     # карты/счета
    (re.compile(r"\+?\d[\d\-\s\(\)]{8,}\d"), "[PHONE]"),          # телефоны без метки
    (re.compile(r"[\w\.-]+@[\w\.-]+"), "[EMAIL]"),                # e-mail
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),         # IPv4
    # ФИО: 2–3 слова на кириллице/латинице с каз. буквами
    (re.compile(rf"\b[{UP}][{LW}\-]+(?:\s+[{UP}][{LW}\-]+){{1,2}}\b"), "[NAME]"),
    # ФИО UPPERCASE (ИВАНОВ ИВАН/ИВАНОВИЧ)
    (re.compile(rf"\b[{UP}]{{2,}}(?:\s+[{UP}]{{2,}}){{1,2}}\b"), "[NAME]"),
]

# 3) Ключи, которые всегда маскируем как персонал/авторы/исполнители
STAFF_KEYS = {
    "assignee", "reporter", "author", "owner", "creator",
    "createdBy", "updatedBy", "resolvedBy", "assigned_to",
}

def anonymize_text(s: str) -> str:
    if not s:
        return s
    for rx, repl in FIELD_PATTERNS:
        s = rx.sub(repl, s)
    for rx, repl in COMMON_PATTERNS:
        s = rx.sub(repl, s)
    return s

def anonymize_any(x: Any, key_hint: str | None = None) -> Any:
    """
    Рекурсивно проходит по объекту JSON и маскирует:
      - любые строки (через anonymize_text)
      - значения по ключам из STAFF_KEYS -> '[STAFF]'
    """
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in STAFF_KEYS and isinstance(v, (str, dict, list)):
                out[k] = "[STAFF]"
            else:
                out[k] = anonymize_any(v, k)
        return out
    if isinstance(x, list):
        return [anonymize_any(v, key_hint) for v in x]
    if isinstance(x, str):
        # если это явно staff-поле, жёстко затираем
        if key_hint in STAFF_KEYS:
            return "[STAFF]"
        return anonymize_text(x)
    # числа/булевы/None не трогаем
    return x

def main():
    src = Path("data/raw")
    dst = Path("data/anonymized")
    dst.mkdir(parents=True, exist_ok=True)

    for fn in src.glob("*.json"):
        with open(fn, encoding="utf-8") as f:
            data = json.load(f)

        anon = anonymize_any(data)

        out = dst / fn.name
        with open(out, "w", encoding="utf-8") as f:
            json.dump(anon, f, ensure_ascii=False, indent=2)

        print(f"[OK] {fn.name} → {out}")

if __name__ == "__main__":
    main()