from __future__ import annotations

from api.main import _build_sd_comment


def test_build_sd_comment_has_4_numbered_sections() -> None:
    structured = {
        "description": ["Сбой авторизации во внешнем сервисе [kb:RP1]"],
        "causes": ["Проверить интеграционный шлюз [kb:RP2]"],
        "actions": ["Попросить повторить вход в инкогнито [kb:RP3]"],
        "next_steps": ["Инцидент не массовый, нужна проверка данных систем [kb:RP4]"],
    }

    comment = _build_sd_comment(structured, raw_answer="")

    assert "1) Суть инцидента:" in comment
    assert "2) Что проверить в системах:" in comment
    assert "3) Рекомендации по действиям клиенту:" in comment
    assert "4) Общий вывод:" in comment
    assert comment.count("\n") == 3
