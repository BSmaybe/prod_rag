import os

from retriever import ingest_new


def _row(solution_text: str | None) -> dict:
    return {
        "issue_key": "RP123",
        "text": "Проблема в приложении, не проходит платеж.",
        "solution_text": solution_text,
        "service": "Мобильные приложения",
    }


def test_filter_short_solution(monkeypatch):
    monkeypatch.setenv("KB_MIN_SOLUTION_CHARS", "30")
    monkeypatch.setenv("KB_REQUIRE_SOLUTION", "1")
    monkeypatch.setenv("KB_STOPWORDS", "")
    monkeypatch.setenv("KB_DEDUP_ENABLED", "0")

    rows, counts = ingest_new._filter_kb_rows([_row("ок")], search_fn=lambda q, top_k=1: [])
    assert rows == []
    assert counts["skipped_short_solution"] == 1


def test_filter_stopword_solution(monkeypatch):
    monkeypatch.setenv("KB_MIN_SOLUTION_CHARS", "1")
    monkeypatch.setenv("KB_REQUIRE_SOLUTION", "1")
    monkeypatch.setenv("KB_STOPWORDS", "решено,сделано,ок")
    monkeypatch.setenv("KB_DEDUP_ENABLED", "0")

    rows, counts = ingest_new._filter_kb_rows([_row("Ок")], search_fn=lambda q, top_k=1: [])
    assert rows == []
    assert counts["skipped_stopword"] == 1


def test_filter_missing_solution(monkeypatch):
    monkeypatch.setenv("KB_REQUIRE_SOLUTION", "1")
    monkeypatch.setenv("KB_DEDUP_ENABLED", "0")

    rows, counts = ingest_new._filter_kb_rows([_row("")], search_fn=lambda q, top_k=1: [])
    assert rows == []
    assert counts["skipped_no_solution"] == 1


def test_filter_dedup(monkeypatch):
    monkeypatch.setenv("KB_MIN_SOLUTION_CHARS", "1")
    monkeypatch.setenv("KB_REQUIRE_SOLUTION", "1")
    monkeypatch.setenv("KB_STOPWORDS", "")
    monkeypatch.setenv("KB_DEDUP_ENABLED", "1")
    monkeypatch.setenv("KB_DEDUP_SCORE", "0.92")
    monkeypatch.setenv("KB_DEDUP_TOP_K", "1")

    def _hits(_q: str, _k: int = 1):
        return [{"score": 0.95}]

    rows, counts = ingest_new._filter_kb_rows([_row("Решение: проверить логи и перезапустить")], search_fn=_hits)
    assert rows == []
    assert counts["skipped_dedup"] == 1
