# api/manage.py
from fastapi import APIRouter

from retriever.ingest_new import ingest_new_tickets
from retriever.ingest_articles import ingest_articles

router = APIRouter(prefix="/manage", tags=["manage"])


@router.post("/reindex")
def reindex():
    """
    Ручной триггер доиндексации новых тикетов из CSV.
    """
    added = ingest_new_tickets()
    return {"status": "ok", "added": added}


@router.post("/reindex_articles")
def reindex_articles():
    """
    Ручной триггер доиндексации статей из папки ARTICLES_DIR.
    """
    added = ingest_articles()
    return {"status": "ok", "added": added}
