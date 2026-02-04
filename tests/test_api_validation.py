"""Lightweight FastAPI validation checks."""

from __future__ import annotations

import pytest

pytest.importorskip("httpx", reason="httpx is required for TestClient")
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_ask_rejects_empty_issue_text() -> None:
    """The /ask endpoint should reject empty issue_text early."""

    response = client.post("/ask", data={"issue_text": "   "})

    assert response.status_code == 400
    assert response.json()["error"] == "empty_issue_text"
