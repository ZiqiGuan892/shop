#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/sft.sh [training options...]

Run full-parameter SFT with the active Python environment (or PYTHON_BIN).
BASE_MODEL           Local model directory or Hugging Face model ID
                     (default: ../models/Qwen3.5-2B beside the repository)
SFT_TRAIN_FILE       Default: outputs/sft-collection/clean/train.jsonl
SFT_VALIDATION_FILE  Default: outputs/sft-collection/clean/validation.jsonl
SFT_OUTPUT_DIR       Default: outputs/models/sft-full (complete model; no merge)
SFT_NPROC_PER_NODE   Default: 8 (single-node torchrun/DDP); set 1 for one device

Default parallelism: 8 data-parallel replicas, no tensor parallelism.
Each device holds the full training state unless --deepspeed configures sharding.

Extra options go to scripts/train_sft.py, e.g. --learning-rate 2e-5,
--epochs 3, --resume-from-checkpoint PATH, --deepspeed CONFIG.json.
Use the SFT_* variables above to change input/output files.
HELP
  exit 0
fi

for argument in "$@"; do
  case "$argument" in
    --finetuning-type|--finetuning-type=*|--qlora|--qlora=*|--train|--train=*|--validation|--validation=*|--output|--output=*)
      echo "sft.sh uses full fine-tuning; configure files through SFT_TRAIN_FILE, SFT_VALIDATION_FILE and SFT_OUTPUT_DIR." >&2
      exit 2
      ;;
  esac
done

BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT:-$ROOT/../models}/Qwen3.5-2B}"
TRAIN_FILE="${SFT_TRAIN_FILE:-$ROOT/outputs/sft-collection/clean/train.jsonl}"
VALIDATION_FILE="${SFT_VALIDATION_FILE:-$ROOT/outputs/sft-collection/clean/validation.jsonl}"
OUTPUT_DIR="${SFT_OUTPUT_DIR:-$ROOT/outputs/models/sft-full}"
NPROC="${SFT_NPROC_PER_NODE:-8}"
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "SFT_NPROC_PER_NODE must be a positive integer." >&2
  exit 2
fi

cd "$ROOT"
for data_file in "$TRAIN_FILE" "$VALIDATION_FILE"; do
  if [[ ! -f "$data_file" || ! -s "$data_file" ]]; then
    echo "SFT data file is missing or empty: $data_file" >&2
    echo "Run scripts/clean_sft_data.py first." >&2
    exit 1
  fi
done
source "$ROOT/scripts/lib/python_runtime.sh"
init_project_python "$ROOT" SFT
export PYTHONUNBUFFERED=1
launcher=("$PROJECT_PYTHON")
if (( NPROC > 1 )); then
  launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc-per-node "$NPROC")
fi
echo "SFT mode: full; nodes: 1; processes: $NPROC; complete model output: $OUTPUT_DIR" >&2
exec "${launcher[@]}" scripts/train_sft.py \
  --finetuning-type full \
  --model "$BASE_MODEL" \
  --train "$TRAIN_FILE" \
  --validation "$VALIDATION_FILE" \
  --output "$OUTPUT_DIR" \
  --dtype auto \
  --gradient-checkpointing \
  --attention-implementation sdpa \
  "$@"
