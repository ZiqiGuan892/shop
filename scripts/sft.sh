#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/sft.sh [training options...]

Run full-parameter SFT with the active Python environment (or PYTHON_BIN).
BASE_MODEL           Local model directory or Hugging Face model ID
                     (default: ../models/Qwen3.5-9B beside the repository)
SFT_TRAIN_FILE       Default: data/sft/train.jsonl (checked-in cleaned data)
SFT_VALIDATION_FILE  Default: data/sft/validation.jsonl
SFT_OUTPUT_DIR       Default: outputs/models/sft-full (complete model; no merge)
SFT_NPROC_PER_NODE   Default: 8 (single-node data parallelism + ZeRO-3)

Defaults: BF16, batch size 1 per device, gradient checkpointing, TP=1.
DeepSpeed shards parameters, gradients and optimizer states to reduce peak memory.
Default DeepSpeed config: configs/sft_zero3.json (no offload or quantization).
Sequence length and effective global batch are unchanged (24,576 and 64).

Extra options go to scripts/train_sft.py, e.g. --learning-rate 2e-5,
--epochs 3, --resume-from-checkpoint PATH, --deepspeed CONFIG.json.
Use the SFT_* variables above to change input/output files.
HELP
  exit 0
fi

DEEPSPEED_CONFIG="$ROOT/configs/sft_zero3.json"
training_options=()
while (( $# > 0 )); do
  case "$1" in
    --deepspeed)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
        echo "--deepspeed requires a JSON configuration path." >&2
        exit 2
      fi
      DEEPSPEED_CONFIG="$2"
      shift 2
      ;;
    --deepspeed=*)
      DEEPSPEED_CONFIG="${1#*=}"
      if [[ -z "$DEEPSPEED_CONFIG" ]]; then
        echo "--deepspeed requires a JSON configuration path." >&2
        exit 2
      fi
      shift
      ;;
    --finetuning-type|--finetuning-type=*|--qlora|--qlora=*|--train|--train=*|--validation|--validation=*|--output|--output=*)
      echo "sft.sh uses full fine-tuning; configure files through SFT_TRAIN_FILE, SFT_VALIDATION_FILE and SFT_OUTPUT_DIR." >&2
      exit 2
      ;;
    *)
      training_options+=("$1")
      shift
      ;;
  esac
done

BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT:-$ROOT/../models}/Qwen3.5-9B}"
TRAIN_FILE="${SFT_TRAIN_FILE:-$ROOT/data/sft/train.jsonl}"
VALIDATION_FILE="${SFT_VALIDATION_FILE:-$ROOT/data/sft/validation.jsonl}"
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
    echo "Check data/sft/ or point SFT_TRAIN_FILE and SFT_VALIDATION_FILE to your cleaned data." >&2
    exit 1
  fi
done
if [[ ! -f "$DEEPSPEED_CONFIG" || ! -s "$DEEPSPEED_CONFIG" ]]; then
  echo "DeepSpeed configuration is missing or empty: $DEEPSPEED_CONFIG" >&2
  exit 1
fi
source "$ROOT/scripts/lib/python_runtime.sh"
init_project_python "$ROOT" SFT
export PYTHONUNBUFFERED=1
launcher=("$PROJECT_PYTHON")
if (( NPROC > 1 )); then
  launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc-per-node "$NPROC")
fi
echo "SFT mode: full; nodes: 1; processes: $NPROC; complete model output: $OUTPUT_DIR" >&2
echo "DeepSpeed config: $DEEPSPEED_CONFIG" >&2
echo "SFT data: train=$TRAIN_FILE; validation=$VALIDATION_FILE" >&2
exec "${launcher[@]}" scripts/train_sft.py \
  --finetuning-type full \
  --model "$BASE_MODEL" \
  --train "$TRAIN_FILE" \
  --validation "$VALIDATION_FILE" \
  --output "$OUTPUT_DIR" \
  --dtype bf16 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-checkpointing \
  --attention-implementation sdpa \
  --deepspeed "$DEEPSPEED_CONFIG" \
  "${training_options[@]}"
