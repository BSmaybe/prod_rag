# core/vectordb.py
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# -----------------------
# Logging
# -----------------------
def _parse_log_level() -> int:
    """
    Поддерживаем LOG_LEVEL=info / INFO / 20 и т.п.
    """
    s = (os.getenv("LOG_LEVEL") or "INFO").strip()
    if s.isdigit():
        return int(s)
    return getattr(logging, s.upper(), logging.INFO)


logging.basicConfig(level=_parse_log_level())
logger = logging.getLogger("rag.vectordb")

# -----------------------
# Config
# -----------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_tickets")

EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "/app/model_data")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

# performance knobs
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "64"))  # CPU: 32-64 ok
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "200"))
QDRANT_WAIT = (os.getenv("QDRANT_WAIT", "true").strip().lower() in ("1", "true", "yes"))

# если у тебя named vectors:
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "").strip()  # "" => unnamed vector

# search defaults
DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))

_embedder: Optional[SentenceTransformer] = None
_qdrant: Optional[QdrantClient] = None


# -----------------------
# Clients
# -----------------------
def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model from %s (device=%s)", EMBED_MODEL_PATH, EMBED_DEVICE)
        _embedder = SentenceTransformer(EMBED_MODEL_PATH, device=EMBED_DEVICE)
    return _embedder


# -----------------------
# Helpers
# -----------------------
def _ticket_u64_id(issue_key: str) -> int:
    """
    Qdrant point id должен быть int(u64) или UUID.
    Делаем стабильный u64 из issue_key.
    """
    h = hashlib.sha1(issue_key.encode("utf-8")).digest()  # 20 bytes
    return int.from_bytes(h[:8], "big", signed=False)


def _ensure_collection(vector_dim: int) -> None:
    """
    ВАЖНО: НЕ пересоздаем коллекцию (recreate/delete), только создаем если её нет.
    """
    client = _get_qdrant()

    try:
        col = client.get_collection(QDRANT_COLLECTION)
        # проверим, что размерность совпадает (минимальная защита от 400)
        try:
            if QDRANT_VECTOR_NAME:
                cfg = col.config.params.vectors
                # vectors может быть dict (named) или VectorParams
                if isinstance(cfg, dict) and QDRANT_VECTOR_NAME in cfg:
                    existing_dim = int(cfg[QDRANT_VECTOR_NAME].size)
                    if existing_dim != vector_dim:
                        raise RuntimeError(
                            f"Qdrant collection '{QDRANT_COLLECTION}' vector size mismatch: "
                            f"{existing_dim} != {vector_dim} (vector name='{QDRANT_VECTOR_NAME}')"
                        )
            else:
                cfg = col.config.params.vectors
                # unnamed vector: VectorParams
                if hasattr(cfg, "size"):
                    existing_dim = int(cfg.size)
                    if existing_dim != vector_dim:
                        raise RuntimeError(
                            f"Qdrant collection '{QDRANT_COLLECTION}' vector size mismatch: "
                            f"{existing_dim} != {vector_dim}"
                        )
        except Exception:
            # если структура ответа отличается в разных версиях клиента — не валим сервис,
            # но хотя бы логнем
            logger.warning("Could not validate vector dim for existing collection (skipping check).")

        return
    except Exception:
        pass

    # create missing collection
    logger.warning("Qdrant collection '%s' is missing. Creating...", QDRANT_COLLECTION)

    if QDRANT_VECTOR_NAME:
        vectors_config = {QDRANT_VECTOR_NAME: qm.VectorParams(size=vector_dim, distance=qm.Distance.COSINE)}
    else:
        vectors_config = qm.VectorParams(size=vector_dim, distance=qm.Distance.COSINE)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=vectors_config,
    )


def _encode_passages(texts: List[str], *, show_progress: bool = False) -> np.ndarray:
    """
    Батчевое кодирование.
    Возвращает np.ndarray shape=(N, dim)
    """
    emb = _get_embedder().encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(emb, dtype=np.float32)


def _build_index_text(r: Dict[str, str]) -> str:
    """
    Текст, по которому ищем похожее.
    """
    parts: List[str] = []
    svc = (r.get("service") or "").strip()
    if svc:
        parts.append(f"[SERVICE] {svc}")
    t = (r.get("text") or "").strip()
    if t:
        parts.append(t)
    sol = (r.get("solution_text") or "").strip()
    if sol:
        parts.append(f"[SOLUTION] {sol}")
    return "\n".join(parts).strip()


def _encode_query(query: str) -> List[float]:
    """
    Эмбеддинг для запроса (одна строка).
    """
    q = (query or "").strip()
    if not q:
        return []
    vec = _encode_passages([q], show_progress=False)
    return vec[0].tolist()


def _vector_payload(points: List[qm.ScoredPoint]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in points:
        payload = p.payload or {}
        out.append(
            {
                "score": float(p.score),
                "issue_key": payload.get("issue_key"),
                "text": payload.get("text"),
                "solution_text": payload.get("solution_text"),
                "service": payload.get("service"),
            }
        )
    return out


# -----------------------
# Public API
# -----------------------
def upsert_tickets(rows: List[Dict[str, str]]) -> int:
    """
    rows: [{"issue_key": "...", "text": "...", "solution_text": "...", "service": "..."}]
    """
    if not rows:
        return 0

    indexed_texts: List[str] = []
    ids: List[int] = []
    payloads: List[Dict[str, Any]] = []

    for r in rows:
        issue_key = (r.get("issue_key") or "").strip()
        text = (r.get("text") or "").strip()
        if not issue_key or not text:
            continue

        idx_text = _build_index_text(r)
        indexed_texts.append(idx_text)
        ids.append(_ticket_u64_id(issue_key))

        payloads.append(
            {
                "issue_key": issue_key,
                "text": text,
                "solution_text": (r.get("solution_text") or "").strip(),
                "service": (r.get("service") or "").strip(),
            }
        )

    if not indexed_texts:
        return 0

    # encode in one go
    vectors = _encode_passages(indexed_texts, show_progress=True)
    dim = int(vectors.shape[1])

    # ensure collection exists
    _ensure_collection(dim)

    client = _get_qdrant()
    total = 0

    for i in range(0, len(payloads), UPSERT_BATCH_SIZE):
        batch_ids = ids[i : i + UPSERT_BATCH_SIZE]
        batch_vec = vectors[i : i + UPSERT_BATCH_SIZE]
        batch_payload = payloads[i : i + UPSERT_BATCH_SIZE]

        if QDRANT_VECTOR_NAME:
            points = [
                qm.PointStruct(
                    id=batch_ids[j],
                    vector={QDRANT_VECTOR_NAME: batch_vec[j].tolist()},
                    payload=batch_payload[j],
                )
                for j in range(len(batch_ids))
            ]
        else:
            points = [
                qm.PointStruct(
                    id=batch_ids[j],
                    vector=batch_vec[j].tolist(),
                    payload=batch_payload[j],
                )
                for j in range(len(batch_ids))
            ]

        try:
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points,
                wait=QDRANT_WAIT,
            )
            total += len(points)
            logger.info("upsert OK: %s/%s", total, len(payloads))
        except Exception:
            # покажем контекст для 400
            first_key = batch_payload[0].get("issue_key") if batch_payload else None
            logger.exception("upsert FAILED at batch=%s first_issue_key=%s", i // UPSERT_BATCH_SIZE, first_key)
            raise

    return total

def search_hits(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Совместимый поиск по версиям qdrant-client:
    - новые: client.query_points(...)
    - некоторые: client.search_points(...)
    - (старые search() у тебя отсутствует)
    """
    query = (query or "").strip()
    if not query:
        return []

    vec = _encode_query(query)
    if not vec:
        return []

    client = _get_qdrant()

    try:
        # 1) NEW API: query_points
        if hasattr(client, "query_points"):
            if QDRANT_VECTOR_NAME:
                qr = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=qm.NamedVector(name=QDRANT_VECTOR_NAME, vector=vec),
                    limit=top_k,
                    with_payload=True,
                )
            else:
                qr = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=vec,
                    limit=top_k,
                    with_payload=True,
                )
            points = getattr(qr, "points", qr)
            return _vector_payload(points)

        # 2) ALT API: search_points
        if hasattr(client, "search_points"):
            if QDRANT_VECTOR_NAME:
                sr = client.search_points(
                    collection_name=QDRANT_COLLECTION,
                    vector=qm.NamedVector(name=QDRANT_VECTOR_NAME, vector=vec),
                    limit=top_k,
                    with_payload=True,
                )
            else:
                sr = client.search_points(
                    collection_name=QDRANT_COLLECTION,
                    vector=vec,
                    limit=top_k,
                    with_payload=True,
                )
            points = getattr(sr, "result", sr)
            return _vector_payload(points)

        raise RuntimeError(
            "Your qdrant-client has neither query_points nor search_points. "
            "Please upgrade qdrant-client or use REST fallback."
        )

    except Exception:
        logger.exception("search_hits failed")
        raise



def healthcheck_qdrant() -> Dict[str, Any]:
    """
    Быстрая диагностика для /readyz если нужно.
    """
    client = _get_qdrant()
    try:
        col = client.get_collection(QDRANT_COLLECTION)
        return {
            "ok": True,
            "collection": QDRANT_COLLECTION,
            "vectors": str(col.config.params.vectors),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
