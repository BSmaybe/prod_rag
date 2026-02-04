# utils/models.py
from sentence_transformers import SentenceTransformer

# Модели, которые требуют trust_remote_code
_NEED_TRUST = ("gte-", "jina-embeddings-v3", "bge-m3")


def needs_trust(model_name: str) -> bool:
    m = model_name.lower()
    return any(tag in m for tag in _NEED_TRUST)


def load_st_model(model_name: str, device: str = "cpu") -> SentenceTransformer:
    kwargs = {}
    if needs_trust(model_name):
        kwargs["trust_remote_code"] = True
    return SentenceTransformer(model_name, device=device, **kwargs)


def query_prefix(model_name: str, q: str) -> str:
    m = model_name.lower()
    if "e5" in m or "gte-" in m or "jina-embeddings-v3" in m:
        return "query: " + q
    if "bge-m3" in m:
        return "Represent this sentence for searching relevant passages: " + q
    return q


def doc_prefix(model_name: str) -> str:
    m = model_name.lower()
    if "e5" in m or "gte-" in m or "jina-embeddings-v3" in m:
        return "passage: "
    if "bge-m3" in m:
        return "Represent this sentence for retrieval: "
    return ""
