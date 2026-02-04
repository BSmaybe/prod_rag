"""Regression checks for anonymized data sources."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Iterator, Tuple

import pytest

try:  # pragma: no cover - optional dependency
    import psycopg  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psycopg = None


DB_URL = os.getenv("DB_URL")
ANONYMIZED_DIR = Path("data/anonymized")


RE_IIN = re.compile(r"\b\d{12}\b")
RE_EMAIL = re.compile(r"[\w\.-]+@[\w\.-]+")
RE_PHONE = re.compile(r"\+?\d[\d\s\-\(\)]{8,}\d")
RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
RE_FIO = re.compile(
    r"\b[А-ЯЁ][а-яё\-]+ [А-ЯЁ][а-яё\-]+(?: [А-ЯЁ][а-яё\-]+)?\b"
)


def bad_line(text: str) -> bool:
    """Return True if the text contains any patterns that resemble PII."""

    return any(
        pattern.search(text)
        for pattern in (RE_IIN, RE_EMAIL, RE_PHONE, RE_IP, RE_FIO)
    )


def iter_strings(value: object, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], str]]:
    """Yield every string contained in *value* together with its location."""

    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_strings(item, path + (str(key),))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from iter_strings(item, path + (str(index),))


def _iter_anonymized_files() -> Iterable[Path]:
    """Return every anonymized JSON file, skipping the suite if none exist."""

    if not ANONYMIZED_DIR.exists():
        pytest.skip("anonymized dataset directory is missing", allow_module_level=True)

    json_files = sorted(ANONYMIZED_DIR.rglob("*.json"))
    if not json_files:
        pytest.skip("no anonymized JSON files found", allow_module_level=True)

    return json_files


def _suspect_rows(payload: object) -> list[Tuple[int, str, str]]:
    """Collect rows containing potential PII within *payload*."""

    offending: list[Tuple[int, str, str]] = []

    if isinstance(payload, list):
        records: Iterable[Tuple[int, object]] = enumerate(payload, start=1)
    else:
        records = [(1, payload)]

    for index, record in records:
        for entry_path, text in iter_strings(record):
            context = ".".join(entry_path) if entry_path else "<root>"
            if bad_line(text):
                offending.append((index, context, text))
                if len(offending) >= 5:
                    return offending

    return offending


def _shorten(text: str, limit: int = 160) -> str:
    """Return a shortened, single-line representation of *text*."""

    single_line = text.replace("\n", "\\n")
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 1] + "…"


def _format_offending(filename: str, offenders: Sequence[Tuple[int, str, str]]) -> str:
    details = "; ".join(
        f"row {index}, field {context}: {_shorten(text)!r}" for index, context, text in offenders
    )
    return f"{filename} содержит потенциальные персональные данные ({details})"


@pytest.mark.parametrize(
    "json_path",
    [
        pytest.param(path, id=path.relative_to(ANONYMIZED_DIR).as_posix())
        for path in _iter_anonymized_files()
    ],
)
def test_anonymized_files_do_not_contain_pii(json_path: Path) -> None:
    """Validate that every anonymized JSON file is free from obvious PII."""

    try:
        with json_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        pytest.fail(f"{json_path}: invalid JSON - {exc}")

    offenders = _suspect_rows(payload)
    assert not offenders, _format_offending(
        json_path.relative_to(ANONYMIZED_DIR).as_posix(), offenders
    )


def test_anonymized_documents_table() -> None:
    """Ensure the documents table is free from obvious PII when the DB is available."""

    if psycopg is None:
        pytest.skip("psycopg not installed")

    if not DB_URL:
        pytest.skip("DB_URL is not set")

    try:
        with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
            cur.execute("SELECT issue_key, text_chunk FROM documents")
            rows = cur.fetchall()
    except psycopg.OperationalError as exc:
        pytest.skip(f"database unavailable: {exc}")

    offending = [issue for issue, chunk in rows if bad_line(chunk or "")]
    assert not offending, f"обнаружены подозрительные записи: {offending[:5]}"
