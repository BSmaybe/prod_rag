#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, argparse, numpy as np, psycopg
from psycopg import sql
from utils.models import load_st_model, query_prefix  # единая точка

def recall_at_k(results, expected, k):
    exp = list(dict.fromkeys(expected))
    res = list(dict.fromkeys(results))
    hits = sum(1 for x in res[:k] if x in exp)
    return hits / len(exp) if exp else 0.0

def precision_at_k(results, expected, k):
    if k == 0:
        return 0.0
    res = list(dict.fromkeys(results))
    exp = set(expected)
    hits = sum(1 for x in res[:k] if x in exp)
    return hits / min(k, len(res)) if res else 0.0

def mrr(results, expected):
    exp = set(expected)
    seen = set()
    rank = 1
    for x in results:
        if x in seen:
            continue
        seen.add(x)
        if x in exp:
            return 1.0 / rank
        rank += 1
    return 0.0

def bench_table(model_name, table, db_url, k, eval_path):
    m = load_st_model(model_name, device="cpu")  # trust_remote_code внутри
    eval_rows = [json.loads(l) for l in open(eval_path, encoding="utf-8")]
    recs, precs, mrrs, lat = [], [], [], []

    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        for row in eval_rows:
            q = row["query"]
            expected = row["expected"]

            # правильный префикс под модель + encode внутри цикла
            q_enc = query_prefix(model_name, q)
            qvec = m.encode([q_enc], normalize_embeddings=True)[0].tolist()

            t0 = time.time()
            cur.execute(
                sql.SQL("""
                    WITH c AS (
                        SELECT issue_key, text_chunk, embedding <=> %s::vector AS dist
                        FROM {}
                    )
                    SELECT issue_key, text_chunk, (1 - dist) AS score
                    FROM c
                    ORDER BY dist
                    LIMIT %s;
                """).format(sql.Identifier(table)),
                (qvec, k)
            )
            rows_db = cur.fetchall()
            results = [r[0] for r in rows_db]
            lat.append(time.time() - t0)
            recs.append(recall_at_k(results, expected, k))
            precs.append(precision_at_k(results, expected, min(3, k)))
            mrrs.append(mrr(results, expected))

    return {
        "model": model_name, "table": table, "k": k,
        "Recall@K": float(np.mean(recs)),
        "Precision@K": float(np.mean(precs)),
        "MRR": float(np.mean(mrrs)),
        "AvgLatencySec": float(np.mean(lat)),
        "N": len(eval_rows),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("DB_URL"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eval", default="eval/queries.jsonl")
    ap.add_argument("--models", nargs="+", required=True, help="Pairs: model::table")
    args = ap.parse_args()
    if not args.db:
        ap.error("DB_URL must be set via --db or DB_URL environment variable")

    rows = []
    for pair in args.models:
        model, table = pair.split("::", 1)
        rows.append(bench_table(model, table, args.db, args.k, args.eval))

    hdr = ["model","table","k","Recall@K","Precision@K","MRR","AvgLatencySec","N"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"]*len(hdr)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in hdr) + " |")

if __name__ == "__main__":
    main()
