import os
import json
import time
import hashlib
from pathlib import Path

import psycopg
from sentence_transformers import SentenceTransformer

# читаем из окружения; внутри docker хост БД = "postgres"
DB_URL = os.getenv("DB_URL")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
ANON_DIR = Path("data/anonymized")


def chunk(text: str, size=1800, overlap=250):
    """Режем текст на куски, чтобы эмбеддинг не был слишком длинным."""
    if not text:
        return
    words = text.split()
    if not words:
        return
    i = 0
    while i < len(words):
        yield " ".join(words[i:i + size])
        i += max(1, size - overlap)


def ensure_schema(conn, dim: int):
    with conn.cursor() as c:
        # расширение
        c.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # таблица
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
              id         bigserial PRIMARY KEY,
              issue_key  text,
              service    text,
              snippet    text,
              text_chunk text,
              text_hash  text UNIQUE,
              embedding  vector({dim}),
              created_at timestamptz DEFAULT now()
            );
        """)

        # индексы по бизнес-полям
        c.execute("CREATE INDEX IF NOT EXISTS idx_documents_issue_key ON documents(issue_key);")

        # векторный индекс (можно оставить так)
        c.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON documents
            USING ivfflat (embedding vector_l2_ops)
            WITH (lists = 100);
        """)


def build_text(record: dict) -> str:
    """Собираем единый текст из разных возможных полей."""
    parts = []
    for k in ("text", "summary", "description", "resolution"):
        v = record.get(k)
        if v:
            parts.append(str(v))
    return " ".join(parts).strip()


def make_snippet(s: str, n=220) -> str:
    if not s:
        return s
    s = " ".join(s.split())
    return s[:n]


def main():
    if not DB_URL:
        raise RuntimeError("DB_URL environment variable is required")
    print(f"[INFO] loading model: {EMB_MODEL}")
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    dim = model.get_sentence_embedding_dimension()

    with psycopg.connect(DB_URL, autocommit=True) as conn:
        ensure_schema(conn, dim)

        files = sorted(ANON_DIR.glob("*.json"))
        if not files:
            print(f"[WARN] В {ANON_DIR} нет файлов *.json")
            return

        total_chunks = 0

        for fp in files:
            t0 = time.time()
            inserted = 0

            with open(fp, encoding="utf-8") as f:
                arr = json.load(f)

            with conn.cursor() as cur:
                for r in arr:
                    text_full = build_text(r)
                    if not text_full:
                        continue

                    snip = make_snippet(text_full)

                    for ch in chunk(text_full):
                        emb = model.encode([ch], normalize_embeddings=True)[0].tolist()
                        h = hashlib.sha256(ch.encode("utf-8")).hexdigest()

                        cur.execute(
                            """
                            INSERT INTO documents(issue_key, service, snippet, text_chunk, text_hash, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (text_hash) DO NOTHING;
                            """,
                            (
                                r.get("issue_key"),
                                r.get("service"),
                                snip,
                                ch,
                                h,
                                emb,
                            )
                        )
                        inserted += 1
                        total_chunks += 1

            dt = time.time() - t0
            print(f"[OK] {fp.name}: вставлено {inserted} чанков за {dt:.1f}s")

        print(f"[DONE] всего вставлено чанков: {total_chunks}")


if __name__ == "__main__":
    main()
