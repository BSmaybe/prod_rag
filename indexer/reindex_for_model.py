#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, psycopg, argparse
from sentence_transformers import SentenceTransformer
from pathlib import Path
from utils.models import load_st_model, doc_prefix

DB_URL = os.environ.get("DB_URL")

# --- add below imports ---
def _doc_prefix(model_name: str) -> str:
    m = model_name.lower()
    if "e5" in m or "gte-" in m or "jina-embeddings-v3" in m:
        return "passage: "
    if "bge-m3" in m:
        return "Represent this sentence for retrieval: "
    return ""

def ensure_table(conn, model_name, dim):
    tbl = f"documents_{model_name.replace('/','_').replace('-','_')}_{dim}"
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {tbl} (
        id BIGSERIAL PRIMARY KEY,
        source TEXT,
        issue_key TEXT,
        type TEXT,
        service TEXT,
        component TEXT,
        env TEXT,
        version TEXT,
        dt TIMESTAMP,
        text_chunk TEXT,
        embedding VECTOR({dim}),
        metadata JSONB
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
        cur.execute(f"""
          CREATE INDEX IF NOT EXISTS idx_{tbl}_embedding
          ON {tbl} USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);""")
        conn.commit()
    return tbl

def reindex(model_name):
    if not DB_URL:
        raise RuntimeError("DB_URL environment variable is required")
    print(f"[INFO] Using model: {model_name}")
    try:
        m = SentenceTransformer(args.model, device="cpu", trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Failed to load with trust_remote_code for {model_name}: {e}")
        raise
    dim = m.get_sentence_embedding_dimension()
    conn = psycopg.connect(DB_URL)
    tbl = ensure_table(conn, model_name, dim)
    cur = conn.cursor()
    cur.execute("SELECT id, text_chunk, issue_key FROM documents WHERE text_chunk IS NOT NULL LIMIT 5000;")
    rows = cur.fetchall()
    print(f"[INFO] Reindexing {len(rows)} records into {tbl}")
    for i, (id_, text, issue) in enumerate(rows, 1):
        model = load_st_model(model_name, device="cpu")
        pref = doc_prefix(model_name)
        vec = model.encode([pref + text], normalize_embeddings=True)[0].tolist()
        cur.execute(
            f"INSERT INTO {tbl} (source, issue_key, text_chunk, embedding) VALUES (%s,%s,%s,%s)",
            ('reindex', issue, text, vec)
        )
        if i % 100 == 0:
            conn.commit()
            print(f"  -> {i} done")
    conn.commit()
    cur.close(); conn.close()
    print(f"[DONE] {tbl}")
    return tbl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    reindex(args.model)
