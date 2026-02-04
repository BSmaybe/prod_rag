from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from typing import Optional, List, Dict, Iterable, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qexc
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from etl.anonymize import anonymize_text
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
    cfg = os.path.join(path, "config.json")
    if os.path.isdir(path) and not os.path.isfile(cfg):
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

        dim = _embedder.get_sentence_embedding_dimension()
        expected = int(QDRANT_VECTOR_SIZE)
        if expected and dim != expected:
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

    last_err: Exception | None = None
    for _ in range(30):
        try:
            client.get_collection(COLLECTION_NAME)
            _collection_ready = True
            return
        except Exception as e:
            last_err = e
            time.sleep(1)

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


def _make_point_id(issue_key: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{issue_key}:{h}"))


def _encode_query(query: str) -> List[float]:
    return _get_embedder().encode(
        f"query: {query}",
        normalize_embeddings=True,
    ).tolist()


def _encode_passage(text: str) -> List[float]:
    return _get_embedder().encode(
        f"passage: {text}",
        normalize_embeddings=True,
    ).tolist()


def _query_points(query_vec: List[float], top_k: int, service: Optional[str] = None):
    flt = None
    if service:
        flt = Filter(
            must=[FieldCondition(key="service", match=MatchValue(value=service))]
        )

    try:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True,
            query_filter=flt,
        )
    except TypeError:
        # backward-compat for older qdrant-client versions
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True,
            filter=flt,
        )


def search_hits(query: str, top_k: int = 5, service: Optional[str] = None) -> List[Tuple[str, str, float]]:
    _ensure_collection()

    query_vec = _encode_query(query)

    try:
        resp = _query_points(query_vec, top_k, service)
        hits = resp.points if resp and resp.points else []
    except qexc.UnexpectedResponse as e:
        if "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
            log.error("Qdrant collection not found: %s", e)
            return []
        raise

    if not hits:
        return []

    out: List[Tuple[str, str, float]] = []
    for hit in hits:
        payload = hit.payload or {}
        out.append(
            (
                str(payload.get("issue_key", "N/A")),
                str(payload.get("text_chunk", "")),
                float(hit.score or 0.0),
            )
        )
    return out


def search_tickets(query: str, top_k: int = 5, service: Optional[str] = None) -> str:
    hits = search_hits(query, top_k=top_k, service=service)
    if not hits:
        return ""

    context_parts: List[str] = [
        f"Тикет {issue_key}:\n{text_chunk}" for issue_key, text_chunk, _ in hits
    ]

    return "\n\n---\n\n".join(context_parts)


def upsert_tickets(records: Iterable[Dict[str, str]], service: Optional[str] = None) -> int:
    rows = [r for r in records if r.get("issue_key") and r.get("text")]
    if not rows:
        return 0

    _ensure_collection()

    points: List[PointStruct] = []
    for r in rows:
        issue_key = str(r.get("issue_key", "")).strip()
        raw_text = str(r.get("text", "")).strip()
        if not issue_key or not raw_text:
            continue

        text = anonymize_text(raw_text)
        if not text:
            continue

        payload = {
            "issue_key": issue_key,
            "text_chunk": text,
            "snippet": text[:300],
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        if service or r.get("service"):
            payload["service"] = service or r.get("service")

        # Any extra string fields must be anonymized before storing in Qdrant.
        for k, v in r.items():
            if k in {"issue_key", "text", "service"}:
                continue
            if isinstance(v, str):
                payload[k] = anonymize_text(v)
            elif isinstance(v, (int, float, bool)) or v is None:
                payload[k] = v

        vec = _encode_passage(text)
        point_id = _make_point_id(issue_key, text)
        points.append(PointStruct(id=point_id, vector=vec, payload=payload))

    if not points:
        return 0

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)
