#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-model}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"

source "$ROOT/scripts/lib/python_runtime.sh"
init_project_python "$ROOT"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
"$PROJECT_PYTHON" scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --progress-interval "${EVAL_PROGRESS_INTERVAL:-15}" \
  --rollout-print-every "${EVAL_ROLLOUT_PRINT_EVERY:-5}"

echo "Evaluation summary saved; building report in $OUTPUT_DIR" >&2
"$PROJECT_PYTHON" scripts/build_eval_report.py --run-dir "$OUTPUT_DIR"
