from __future__ import annotations

import logging
import os
import time
from typing import Optional, List

from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qexc
from qdrant_client.http.models import Distance, VectorParams

from sentence_transformers import SentenceTransformer

from .config import (
    QDRANT_URL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    QDRANT_VECTOR_SIZE,
    QDRANT_DISTANCE,
)

log = logging.getLogger(__name__)

client = QdrantClient(url=QDRANT_URL)
_embedder: Optional[SentenceTransformer] = None
_collection_ready: bool = False


def _validate_model_path(path: str) -> None:
    # SentenceTransformer локальную модель ожидает как минимум с config.json
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        raise RuntimeError(
            f"Embedding model path is invalid: {path}. "
            f"Missing config.json. "
            f"Fix docker volume mount (expected local_model -> /app/model_data)."
        )


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        device = (EMBEDDING_DEVICE or "cpu").lower()
        if device != "cpu":
            device = "cpu"

        _validate_model_path(EMBEDDING_MODEL)
        log.info("Loading embedding model from %s (device=%s)", EMBEDDING_MODEL, device)

        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

        # sanity check: размер должен совпасть с QDRANT_VECTOR_SIZE
        dim = _embedder.get_sentence_embedding_dimension()
        expected = int(QDRANT_VECTOR_SIZE)
        if dim != expected:
            raise RuntimeError(
                f"Embedding dim mismatch: model={dim}, QDRANT_VECTOR_SIZE={expected}. "
                f"Set QDRANT_VECTOR_SIZE={dim} in env/config and recreate collection."
            )

    return _embedder


def _ensure_collection() -> None:
    global _collection_ready
    if _collection_ready:
        return

    dist = Distance.COSINE if (QDRANT_DISTANCE or "Cosine").lower() == "cosine" else Distance.DOT
    vector_size = int(QDRANT_VECTOR_SIZE)

    # Ретраим Qdrant, потому что он может быть health:starting
    last_err: Exception | None = None
    for _ in range(30):  # ~30 секунд
        try:
            client.get_collection(COLLECTION_NAME)
            _collection_ready = True
            return
        except Exception as e:
            last_err = e
            time.sleep(1)

    # Если коллекции нет — попробуем создать (тоже с ретраями)
    for _ in range(10):
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=dist),
            )
            log.warning("Qdrant collection '%s' was missing and has been created.", COLLECTION_NAME)
            _collection_ready = True
            return
        except qexc.UnexpectedResponse as e:
            if "already exists" in str(e).lower():
                _collection_ready = True
                return
            last_err = e
            time.sleep(1)
        except Exception as e:
            last_err = e
            time.sleep(1)

    raise RuntimeError(f"Qdrant is not ready or collection init failed. Last error: {last_err}")


def search_tickets(query: str, top_k: int = 5) -> str:
    _ensure_collection()

    query_vec = _get_embedder().encode(
        f"query: {query}",
        normalize_embeddings=True,
    ).tolist()

    try:
        resp = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )
        hits = resp.points if resp and resp.points else []
    except qexc.UnexpectedResponse as e:
        if "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
            log.error("Qdrant collection not found: %s", e)
            return ""
        raise

    if not hits:
        return ""

    context_parts: List[str] = []
    for hit in hits:
        payload = hit.payload or {}
        part = f"Тикет {payload.get('issue_key', 'N/A')}:\n{payload.get('text_chunk', '')}"
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)
