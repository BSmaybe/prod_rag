from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from core.vectordb import upsert_articles
from etl.anonymize import anonymize_text
from core.config import OLLAMA_URL, OLLAMA_MODEL
import requests

log = logging.getLogger("rag.ingest.articles")

ARTICLES_DIR = os.getenv("ARTICLES_DIR", "/data/articles")
ARTICLES_MAX_CHUNK_CHARS = int(os.getenv("ARTICLES_MAX_CHUNK_CHARS", "900"))
ARTICLES_ANONYMIZE = os.getenv("ARTICLES_ANONYMIZE", "0").strip().lower() in {"1", "true", "yes", "on"}

ARTICLES_JUDGE_ENABLED = os.getenv("ARTICLES_JUDGE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
ARTICLES_JUDGE_MODEL = os.getenv("ARTICLES_JUDGE_MODEL", OLLAMA_MODEL)
ARTICLES_JUDGE_CACHE_DIR = os.getenv("ARTICLES_JUDGE_CACHE_DIR", "data/articles_clean")
ARTICLES_JUDGE_MAX_CHARS = int(os.getenv("ARTICLES_JUDGE_MAX_CHARS", "6000"))
ARTICLES_JUDGE_TIMEOUT_SEC = float(os.getenv("ARTICLES_JUDGE_TIMEOUT_SEC", "60"))
ARTICLES_JUDGE_CONNECT_TIMEOUT_SEC = float(os.getenv("ARTICLES_JUDGE_CONNECT_TIMEOUT_SEC", "10"))
ARTICLES_JUDGE_NUM_CTX = int(os.getenv("ARTICLES_JUDGE_NUM_CTX", "2048"))
ARTICLES_JUDGE_NUM_PREDICT = int(os.getenv("ARTICLES_JUDGE_NUM_PREDICT", "320"))
ARTICLES_JUDGE_MODE = os.getenv("ARTICLES_JUDGE_MODE", "product").strip().lower()

_PARA_SPLIT = re.compile(r"\n\s*\n+")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_title(text: str, fallback: str) -> Tuple[str, str]:
    """
    Возвращает (title, body_without_title).
    Для MD берём первую строку вида '# ...'.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            title = line.lstrip("#").strip()
            body = "\n".join(lines[i + 1 :]).strip()
            return title or fallback, body
    return fallback, text.strip()


def _chunk_paragraphs(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    max_chars = max(200, int(max_chars))
    paras = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append("\n\n".join(cur).strip())
        cur = []
        cur_len = 0

    for p in paras:
        if len(p) > max_chars:
            # если абзац слишком большой — режем
            flush()
            for i in range(0, len(p), max_chars):
                part = p[i : i + max_chars].strip()
                if part:
                    chunks.append(part)
            continue

        if cur_len + len(p) + 2 <= max_chars:
            cur.append(p)
            cur_len += len(p) + 2
        else:
            flush()
            cur.append(p)
            cur_len = len(p)

    flush()
    return [c for c in chunks if c]


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_paths(path: Path) -> Tuple[Path, Path]:
    base = Path(ARTICLES_DIR)
    rel = path.relative_to(base)
    cache_base = Path(ARTICLES_JUDGE_CACHE_DIR)
    clean_path = (cache_base / rel).with_suffix(".clean.txt")
    hash_path = clean_path.with_suffix(clean_path.suffix + ".sha1")
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    return clean_path, hash_path


def _load_cached(clean_path: Path, hash_path: Path, digest: str) -> Optional[str]:
    if not clean_path.exists() or not hash_path.exists():
        return None
    try:
        if hash_path.read_text(encoding="utf-8").strip() == digest:
            return clean_path.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def _save_cache(clean_path: Path, hash_path: Path, digest: str, text: str) -> None:
    try:
        clean_path.write_text(text, encoding="utf-8")
        hash_path.write_text(digest, encoding="utf-8")
    except Exception:
        pass


def _judge_prompt(raw_text: str) -> str:
    if ARTICLES_JUDGE_MODE == "product":
        return f"""Ты — строгий редактор базы знаний по продукту банка.
Твоя задача — привести статью к компактному, полезному виду для поиска.

Сырая статья:
\"\"\"{raw_text}\"\"\"

Правила:
1) Если статья — пустая, рекламная, не про продукт/функции/ограничения — верни строго: REJECT
2) Если статья полезна, перепиши её строго в формате:
ПРОДУКТ: <название/модуль>
НАЗНАЧЕНИЕ: <что делает>
ОСОБЕННОСТИ: <ключевые факты/правила>
ОГРАНИЧЕНИЯ: <что нельзя/лимиты/важные условия>
"""

    # default: incident mode
    return f"""Ты — строгий IT-аудитор базы знаний.
Твоя задача — проверить, содержит ли статья полезную информацию для решения будущих проблем.

Сырая статья:
\"\"\"{raw_text}\"\"\"

Правила:
1) Если НЕТ четкого описания проблемы ИЛИ НЕТ конкретных шагов решения (а только отписки), верни строго: REJECT
2) Если статья полезна, перепиши её строго в формате:
СИМПТОМЫ: <кратко суть>
РЕШЕНИЕ: <конкретные команды, проверки, действия>
"""


def _judge_article(raw_text: str) -> Optional[str]:
    prompt = _judge_prompt(raw_text[:ARTICLES_JUDGE_MAX_CHARS])
    payload = {
        "model": ARTICLES_JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": ARTICLES_JUDGE_NUM_CTX,
            "num_predict": ARTICLES_JUDGE_NUM_PREDICT,
            "stop": ["\n\nREJECT", "\n\nОтвет:", "\n\n---"],
        },
    }

    timeout = (ARTICLES_JUDGE_CONNECT_TIMEOUT_SEC, ARTICLES_JUDGE_TIMEOUT_SEC)
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"judge_http_error {resp.status_code}")

    data = resp.json() if resp.content else {}
    answer = str(data.get("response", "")).strip()
    if not answer:
        return None
    if answer.strip().upper().startswith("REJECT"):
        return None
    return answer.strip()


def _collect_article_rows() -> List[Dict[str, str]]:
    base = Path(ARTICLES_DIR)
    if not base.exists() or not base.is_dir():
        return []

    rows: List[Dict[str, str]] = []
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        raw = _read_text(path)
        title, body = _extract_title(raw, fallback=path.stem)

        text_for_judge = (f"{title}\n\n{body}" if body else title).strip()
        if not text_for_judge:
            continue

        if ARTICLES_JUDGE_ENABLED:
            digest = _sha1_text(text_for_judge)
            clean_path, hash_path = _cache_paths(path)
            cached = _load_cached(clean_path, hash_path, digest)
            if cached is not None:
                judged_text = cached
            else:
                judged_text = _judge_article(text_for_judge)
                if judged_text:
                    _save_cache(clean_path, hash_path, digest, judged_text)
            if not judged_text:
                continue
            body = judged_text

        chunks = _chunk_paragraphs(body, ARTICLES_MAX_CHUNK_CHARS)
        if not chunks:
            continue

        for idx, chunk in enumerate(chunks):
            text = anonymize_text(chunk) if ARTICLES_ANONYMIZE else chunk
            rows.append(
                {
                    "doc_id": path.stem,
                    "title": title,
                    "text": text,
                    "chunk_index": str(idx),
                    "source_path": str(path),
                }
            )
    return rows


def ingest_articles() -> int:
    rows = _collect_article_rows()
    if not rows:
        return 0

    added = upsert_articles(rows)
    log.info("articles_ingest total_rows=%s added=%s", len(rows), added)
    return added
