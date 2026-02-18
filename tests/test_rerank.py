from __future__ import annotations

from api.main import _rerank_hits


def test_rerank_prefers_lexically_relevant_hit() -> None:
    query = "ошибка авторизации пользователя в мобильном приложении"
    hits = [
        {"issue_key": "A-1", "text": "сбой печати отчета в бэк-офисе", "score": 0.93},
        {"issue_key": "A-2", "text": "ошибка авторизации клиента в мобильном приложении", "score": 0.78},
    ]

    reranked = _rerank_hits(query, hits, top_k=1)

    assert len(reranked) == 1
    assert reranked[0]["issue_key"] == "A-2"


def test_rerank_returns_top_k_items() -> None:
    query = "таймаут интеграции"
    hits = [
        {"issue_key": "A-1", "text": "таймаут при вызове интеграции", "score": 0.7},
        {"issue_key": "A-2", "text": "таймаут чтения БД", "score": 0.6},
        {"issue_key": "A-3", "text": "ошибка формата файла", "score": 0.5},
    ]

    reranked = _rerank_hits(query, hits, top_k=2)

    assert len(reranked) == 2
