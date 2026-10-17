#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTOR_CHECKPOINT="${1:?usage: bash scripts/export_grpo.sh <global_step_*/actor> [output_dir]}"
OUTPUT_DIR="${2:-$ROOT/outputs/models/grpo-merged}"

source "$ROOT/scripts/lib/python_runtime.sh"
init_project_python "$ROOT" "GRPO export"
if [[ -n "${VERL_ROOT:-}" ]]; then
  if [[ ! -f "$VERL_ROOT/verl/__init__.py" ]]; then
    echo "VERL_ROOT must point to the checkout containing verl/__init__.py." >&2
    exit 1
  fi
  export PYTHONPATH="$VERL_ROOT:$PYTHONPATH"
fi
exec "$PROJECT_PYTHON" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$OUTPUT_DIR" \
  --trust-remote-code
