# api/manage.py
from fastapi import APIRouter

from retriever.ingest_new import ingest_new_tickets

router = APIRouter(prefix="/manage", tags=["manage"])


@router.post("/reindex")
def reindex():
    """
    Ручной триггер доиндексации новых тикетов из CSV.
    """
    added = ingest_new_tickets()
    return {"status": "ok", "added": added}
