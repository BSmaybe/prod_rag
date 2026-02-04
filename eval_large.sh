#!/usr/bin/env bash
set -euo pipefail

# --- конфиг ---
if [[ -z "${DB_URL:-}" ]]; then
  echo "❌ DB_URL не задан. Укажите его через переменные окружения или .env"
  exit 1
fi
export DB_URL
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:--1}
VENV_PY="./.venv/bin/python"
MODEL="intfloat/multilingual-e5-small"
TABLE="documents_intfloat_multilingual_e5_small_384"
K=5

echo "[ENV] DB_URL=$DB_URL"
echo "[ENV] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# --- проверки окружения ---
if [[ ! -x "$VENV_PY" ]]; then
  echo "❌ Не найден venv Python по пути $VENV_PY"
  echo "   Активируй окружение:  source .venv/bin/activate"
  exit 1
fi

# --- sanity: eval/queries.jsonl (если вдруг нет) ---
if [[ ! -f "eval/queries.jsonl" ]]; then
  echo "[INIT] Создаю eval/queries.jsonl"
  mkdir -p eval
  cat > eval/queries.jsonl << 'EOF'
{"query": "ERR_AUTH_TIMEOUT при оплате", "expected": ["3967657","3957302"]}
{"query": "клиент не может войти", "expected": ["3957302"]}
{"query": "не может совершить платеж", "expected": ["3962104"]}
EOF
fi

# --- переиндексация для модели ---
echo "[INDEX] Reindex for $MODEL -> $TABLE"
$VENV_PY indexer/reindex_for_model.py --model "$MODEL"

# --- бенчмарк модели ---
echo "[BENCH] Запускаю проверку модели при k=$K"
$VENV_PY benchmarks/bench_retrieval.py \
  --db "$DB_URL" \
  --k $K \
  --models \
  "$MODEL::$TABLE"

# --- быстрая дымовая проверка на живых данных ---
echo "[SMOKE] retrieval_smoke на 'ERR_AUTH_TIMEOUT при оплате'"
$VENV_PY tests/retrieval_smoke.py \
  --db "$DB_URL" \
  --model "$MODEL" \
  --table "$TABLE" \
  --q "ERR_AUTH_TIMEOUT при оплате" \
  --k 5 || true

echo "[DONE] Готово."
