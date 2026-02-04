# retriever/hybrid_search.py
from __future__ import annotations

import importlib.util
import os
from typing import Dict, List, Optional, Tuple

import psycopg
from core.vectordb import search_hits as qdrant_search_hits

DB_URL = os.getenv("DB_URL")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
TOP_K_DEFAULT = int(os.getenv("TOP_K", 5))
TABLE = os.getenv("RETR_TABLE", "documents")

_model = None  # ленивое кэширование


def get_db_url() -> str:
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL environment variable is required")
    return db_url


class NoDocumentsError(RuntimeError):
    """Выбрасывается, когда в таблице нет ни одного документа."""



def ensure_schema(conn: psycopg.Connection) -> None:
    """Создаёт таблицу для документов, если её ещё нет."""

    model = get_model()
    dim = int(model.get_sentence_embedding_dimension())

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
              id         bigserial PRIMARY KEY,
              issue_key  text,
              service    text,
              snippet    text,
              text_chunk text,
              text_hash  text UNIQUE,
              embedding  vector({dim}),
              created_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_issue_key ON {TABLE}(issue_key);"
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE}_embedding
            ON {TABLE}
            USING ivfflat (embedding vector_l2_ops)
            WITH (lists = 100);
            """
        )


def _count_documents(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
        res = cur.fetchone()
        return int(res[0]) if res else 0


def get_model():
    """
    Лениво загружаем sentence-transformers, чтобы контейнер стартовал быстро.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMB_MODEL, device="cpu")
    return _model


# Если у тебя уже есть llm/rephrase.py с функцией rephrase_issue — используем её.
# Если нет — можно временно отключить use_rephrase в search().
def _load_rephrase_fn():
    spec = importlib.util.find_spec("llm.rephrase")
    if not spec or not spec.loader:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "rephrase_issue", None)


def _default_rephrase(text: str) -> str:
    return text


rephrase_issue = _load_rephrase_fn() or _default_rephrase


def _search_with_vector(
    conn: psycopg.Connection,
    qvec: List[float],
    limit: int,
    service: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """
    Низкоуровневый SQL-поиск по вектору.
    Возвращает список (issue_key, snippet, score), где score — "чем выше, тем лучше".
    """
    # Базовый SQL: для каждого issue_key берём наиболее близкий chunk по dist
    where_clause = ""
    params: List[object] = [qvec]

    if service:
        where_clause = "WHERE service = %s"
        params.append(service)

    params.append(limit)

    sql = f"""
    WITH c AS (
      SELECT DISTINCT ON (issue_key)
             issue_key,
             LEFT(text_chunk, 300) AS snippet,
             (embedding <-> %s::vector) AS dist
      FROM {TABLE}
      {where_clause}
      ORDER BY issue_key, dist ASC
    )
    SELECT issue_key, snippet, (-dist) AS score
    FROM c
    ORDER BY score DESC
    LIMIT %s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()  # [(issue_key, snippet, score), ...]
        return rows


def search(
    q: str,
    top_k: Optional[int] = None,
    use_rephrase: bool = True,
    service: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """
    Главная функция поиска.

    Возвращает список кортежей:
      (issue_key, snippet, score)

    Параметры:
      q           — текст запроса (описание инцидента);
      top_k       — сколько уникальных issue_key вернуть (по умолчанию TOP_K_DEFAULT);
      use_rephrase — использовать ли переформулировку запроса через LLM;
      service     — опциональный фильтр по полю service (для разных очередей/систем).
    """
    k = int(top_k or TOP_K_DEFAULT)
    k = max(1, min(100, k))  # немного защищаемся от странных значений

    backend = (os.getenv("VECTOR_BACKEND") or "qdrant").lower()
    if backend in {"qdrant", "qd"}:
        return qdrant_search_hits(q, top_k=k, service=service)
    if backend == "both":
        qd = qdrant_search_hits(q, top_k=k, service=service)
        if qd:
            return qd

    model = get_model()

    # Чтобы не потерять хорошие совпадения, запрашиваем из БД чуть больше,
    # а потом уже агрегируем по issue_key.
    k_sql = max(k * 2, k + 5)

    # Вектор для оригинального текста
    qvec_orig = model.encode([q], normalize_embeddings=True)[0].tolist()

    # По желанию — вектор для переформулированного текста
    q_rephrased: Optional[str] = None
    qvec_rephrased: Optional[List[float]] = None

    if use_rephrase:
        q_rephrased = rephrase_issue(q)

        if q_rephrased and q_rephrased.strip() and q_rephrased.strip() != q.strip():
            qvec_rephrased = model.encode([q_rephrased], normalize_embeddings=True)[0].tolist()

    combined: Dict[str, Dict[str, object]] = {}

    with psycopg.connect(get_db_url()) as conn:
        ensure_schema(conn)

        if _count_documents(conn) == 0:
            raise NoDocumentsError(
                "В таблице нет данных для поиска. Запусти индексатор (indexer/embeddings.py),"
                " чтобы загрузить чанки в БД."
            )

        # Поиск по оригинальному запросу
        rows_orig = _search_with_vector(conn, qvec_orig, k_sql, service)

        # Поиск по переформулированному запросу (если есть)
        rows_reph: List[Tuple[str, str, float]] = []
        if qvec_rephrased is not None:
            rows_reph = _search_with_vector(conn, qvec_rephrased, k_sql, service)

    def add_rows(rows: List[Tuple[str, str, float]], key: str) -> None:
        for issue_key, snippet, score in rows:
            item = combined.setdefault(
                issue_key,
                {"snippet": snippet, "score_orig": 0.0, "score_reph": 0.0},
            )
            # Берём самый длинный сниппет как более информативный
            if snippet and len(str(snippet)) > len(str(item["snippet"])):
                item["snippet"] = snippet
            # Сохраняем максимум по каждому каналу
            item[key] = max(float(item[key]), float(score))  # type: ignore[index]

    add_rows(rows_orig, "score_orig")
    if rows_reph:
        add_rows(rows_reph, "score_reph")

    # Собираем финальный результат с комбинированным скором
    results: List[Tuple[str, str, float]] = []
    for issue_key, data in combined.items():
        score_orig = float(data.get("score_orig") or 0.0)
        score_reph = float(data.get("score_reph") or 0.0)

        # Веса можно подкрутить из env позже, пока — 0.6 / 0.4
        final_score = 0.6 * score_orig + 0.4 * score_reph
        results.append((issue_key, str(data["snippet"]), final_score))

    # Сортируем по убыванию score и режем по top_k
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:k]
