#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage: bash scripts/serve_model.sh [MODEL] [vLLM serve options...]

MODEL defaults to Qwen/Qwen3.5-2B. Additional options are passed to vLLM.
Existing model directories are used directly. Otherwise, short names or
organization/name identifiers resolve to MODEL_ROOT/name (local files only).
MODEL_ROOT defaults to the models directory beside this repository.
Launcher selection: PYTHON_BIN override, vllm on PATH, Python with vllm on
PATH, then the repository's .venv/bin/vllm.

To use an existing training/inference environment, activate it first or set:
  PYTHON_BIN=/path/to/env/bin/python bash scripts/serve_model.sh MODEL

Example for eager execution:
  bash scripts/serve_model.sh Qwen/Qwen3.5-4B --enforce-eager

Other environment variables: SERVED_MODEL_NAME (shopping-agent), LLM_PORT (8000).
USAGE
  exit 0
fi

MODEL="${1:-Qwen/Qwen3.5-2B}"
if (( $# > 0 )); then shift; fi
MODEL_ROOT="${MODEL_ROOT:-$ROOT/../models}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
LLM_PORT="${LLM_PORT:-8000}"

# Keep explicit absolute/relative paths intact. Names such as Qwen/Qwen3.5-4B
# refer to the same local checkpoint as Qwen3.5-4B; never download on a typo.
if [[ -d "$MODEL" ]]; then
  model_dir="$MODEL"
elif [[ "$MODEL" == /* || "$MODEL" == ./* || "$MODEL" == ../* || "$MODEL" == */*/* || "$MODEL" == */ ]]; then
  echo "Model directory does not exist: $MODEL" >&2
  exit 1
else
  model_dir="$MODEL_ROOT/${MODEL##*/}"
fi
if [[ ! -d "$model_dir" ]]; then
  echo "Local model directory does not exist: $model_dir" >&2
  echo "Set MODEL_ROOT to the directory containing your models, or pass an existing model directory." >&2
  exit 1
fi
MODEL="$(cd "$model_dir" && pwd)"
printf 'Using local model: %s\n' "$MODEL" >&2

launcher=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! python_bin="$(command -v "$PYTHON_BIN")"; then
    echo "PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
    exit 1
  fi
  launcher=("$python_bin" -m vllm.entrypoints.cli.main)
elif vllm_bin="$(command -v vllm)"; then
  launcher=("$vllm_bin")
else
  # A package can be installed without its console script being on PATH.
  # Only check module discovery here; real startup/import errors stay visible.
  for candidate in python python3; do
    if python_bin="$(command -v "$candidate")" &&
      "$python_bin" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("vllm") is None)' 2>/dev/null; then
      launcher=("$python_bin" -m vllm.entrypoints.cli.main)
      break
    fi
  done
  if (( ${#launcher[@]} == 0 )) && [[ -x "$ROOT/.venv/bin/vllm" ]]; then
    launcher=("$ROOT/.venv/bin/vllm")
  fi
fi

if (( ${#launcher[@]} == 0 )); then
  cat >&2 <<'ERROR'
Cannot find a vLLM launcher on PATH or in the repository's .venv/bin/vllm.
Activate your existing vLLM environment, or set PYTHON_BIN=/path/to/env/bin/python.
Check that same interpreter with: /path/to/env/bin/python -m pip show vllm
This does not mean vLLM is absent from other Python environments.
ERROR
  exit 1
fi

printf 'Using vLLM launcher:' >&2
printf ' %q' "${launcher[@]}" >&2
printf '\n' >&2

exec "${launcher[@]}" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$LLM_PORT" \
  --max-model-len 24576 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
