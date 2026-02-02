# api/utils/postprocess.py
from __future__ import annotations
from typing import Dict, List
import re

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = re.sub(r"\s+", " ", x.strip().rstrip(" ;")).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x.strip().rstrip(" ;"))
    return out

def postprocess(s: Dict[str, List[str] | str]) -> Dict[str, List[str] | str]:
    desc = list(s.get("description") or [])
    causes = list(s.get("causes") or [])
    actions = list(s.get("actions") or [])
    next_steps = list(s.get("next_steps") or [])

    # 1) Перенос «Рекомендуемые …» из причин в действия
    moved = []
    remain = []
    for c in causes:
        if re.search(r"(рекомендуемые\s+действия|шаги|сделать|проверить|перезапустить|обновить)", c, re.I):
            moved.append(c)
        else:
            remain.append(c)
    causes = remain
    actions.extend(moved)

    # 2) Чистка/дедуп
    desc = _dedupe_keep_order(desc)[:3]
    causes = _dedupe_keep_order(causes)[:6]
    actions = _dedupe_keep_order(actions)[:8]
    next_steps = _dedupe_keep_order(next_steps)[:6]

    # 3) Если пусто — добавим базовые шаги
    if not actions:
        actions = [
            "Проверить версию приложения и обновить до последней.",
            "Проверить тип сети (Wi-Fi/4G) и качество соединения; попробовать альтернативную сеть.",
            "Проверить статус блокировок/ограничений клиента в АБС и снять при наличии оснований.",
            "Сбросить сессию/токены для клиента; повторный вход.",
            "Очистить кеш/данные приложения и повторно авторизоваться.",
        ]

    if not next_steps:
        next_steps = [
            "Если не помогло — собрать: ФИО/ИИН, номер телефона, устройство/OS, версия приложения, время и скрин/код ошибки, тип сети.",
            "Эскалировать в 2-ю линию/разработку с собранными артефактами и ID клиента.",
        ]

    s["description"] = desc
    s["causes"] = causes
    s["actions"] = actions
    s["next_steps"] = next_steps
    return s
