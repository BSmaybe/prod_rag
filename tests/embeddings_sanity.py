#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, math, argparse
from sentence_transformers import SentenceTransformer
import numpy as np

def cos(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    m = SentenceTransformer(args.model, device="cpu")
    dim = m.get_sentence_embedding_dimension()
    print(f"[MODEL] {args.model} dim={dim}")

    # Пары близких / далёких фраз
    pairs = [
        ("ошибка авторизации", "таймаут входа"),
        ("ошибка авторизации", "успешная оплата"),
        ("ERR_AUTH_TIMEOUT при оплате", "500 ошибка на /api/pay"),
        ("ERR_AUTH_TIMEOUT при оплате", "настройка профиля пользователя"),
    ]

    for a, b in pairs:
        va = m.encode([a], normalize_embeddings=True)[0]
        vb = m.encode([b], normalize_embeddings=True)[0]
        c = cos(va, vb)
        print(f"cos('{a}' , '{b}') = {c:.3f}")

    # Ожидание: cos(похожих) > cos(разных)
    print("\n[OK] Санити-проверка пройдена, если похожие пары > 0.5, разные < 0.4 (примерные пороги).")

if __name__ == "__main__":
    main()
