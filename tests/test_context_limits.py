"""Unit checks for LLM context limiting helper."""

from __future__ import annotations

from api.main import _build_limited_context


def test_context_limits_trim_each_chunk() -> None:
    info = _build_limited_context(
        ["x" * 40],
        max_chunk_chars=12,
        max_context_chars=200,
        min_tail_chars=4,
    )

    assert info["chunks_in"] == 1
    assert info["chunks_used"] == 1
    assert info["context"] == "x" * 12
    assert info["used_chars"] == 12


def test_context_limits_total_chars_respected() -> None:
    info = _build_limited_context(
        ["a" * 10, "b" * 10, "c" * 10],
        max_chunk_chars=10,
        max_context_chars=25,
        min_tail_chars=4,
    )

    assert info["used_chars"] <= 25
    assert len(info["context"]) <= 25
    assert info["chunks_used"] == 2


def test_context_limits_tail_fill_on_overflow() -> None:
    info = _build_limited_context(
        ["a" * 10, "b" * 20],
        max_chunk_chars=20,
        max_context_chars=22,
        min_tail_chars=5,
    )

    assert info["used_chars"] == 22
    assert info["context"].endswith("b" * 5)


def test_context_limits_fallback_to_first_chunk() -> None:
    info = _build_limited_context(
        ["z" * 30, "q" * 30],
        max_chunk_chars=30,
        max_context_chars=10,
        min_tail_chars=20,  # специально не даём вставить хвост в основном цикле
    )

    assert info["chunks_in"] == 2
    assert info["chunks_used"] == 1
    assert info["context"] == "z" * 10
    assert info["used_chars"] == 10


def test_context_limits_ignore_empty_noise() -> None:
    info = _build_limited_context(
        ["   ", "\n\t", " useful   text  "],
        max_chunk_chars=50,
        max_context_chars=50,
        min_tail_chars=5,
    )

    assert info["chunks_in"] == 1
    assert info["chunks_used"] == 1
    assert info["context"] == "useful text"
