#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, psycopg, time
from sentence_transformers import SentenceTransformer
from utils.models import load_st_model, query_prefix

def _qpref(model_name: str, q: str) -> str:
    m = model_name.lower()
    if "e5" in m or "gte-" in m or "jina-embeddings-v3" in m:
        return "query: " + q
    if "bge-m3" in m:
        return "Represent this sentence for searching relevant passages: " + q
    return q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    print(f"QUERY: {args.q}")
    print(f"TABLE: {args.table} MODEL: {args.model}")

    m = load_st_model(args.model, device="cpu")
    qvec = m.encode([query_prefix(args.model, args.q)], normalize_embeddings=True)[0].tolist()

    with psycopg.connect(args.db) as conn, conn.cursor() as cur:
        t0 = time.time()
        cur.execute(
            f"""
            WITH dists AS (
              SELECT issue_key, text_chunk, embedding <=> %s::vector AS dist
              FROM {args.table}
            )
            SELECT issue_key,
                   1 - MIN(dist) AS score,
                   (ARRAY_AGG(text_chunk ORDER BY dist))[1] AS text_chunk
            FROM dists
            GROUP BY issue_key
            ORDER BY MIN(dist)
            LIMIT %s;
            """,
            (qvec, args.k)
        )
        for r in cur.fetchall():
            print(f"{r[0]:<10} score={r[1]:.3f}  {r[2][:160]}")
    print(f"[DONE] latency={(time.time() - t0):.3f}s")

if __name__ == "__main__":
    main()
