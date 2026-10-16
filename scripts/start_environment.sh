#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/start_environment.sh

Start the installed ShopSimulator service in the foreground; Ctrl+C stops it.
Install it with: bash scripts/setup.sh --shopsim-only
This command does not install or start veRL, Ray or a model server.

SHOPSIM_PORT       Service port (default: 5700)
SHOPSIM_ENV_SLOTS  Concurrent environment slots (default: 8; not the NPU count)
SHOP_MAX_STEPS     Steps per episode (default: 35)
SHOPSIM_LOG_DIR    Console log directory (default: outputs/shopsimulator)
HELP
  exit 0
fi
if (( $# > 0 )); then
  echo "Unexpected arguments. Use: bash scripts/start_environment.sh --help" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/environments/ShopSimulator/.venv-shopsim"
LOG_DIR="${SHOPSIM_LOG_DIR:-$ROOT/outputs/shopsimulator}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
: >> "$LOG_DIR/server.log"
exec > >(tee -a "$LOG_DIR/server.log") 2>&1

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "ShopSimulator is not installed. Run: bash scripts/setup.sh --shopsim-only" >&2
  exit 1
fi

export SHOPSIM_PYTHON_BIN="$ENV_DIR/bin/python"
export PYTHONUNBUFFERED=1
export PATH="$ENV_DIR/bin:$PATH"
echo "ShopSimulator Python: $SHOPSIM_PYTHON_BIN"
echo "Environment API: http://127.0.0.1:${SHOPSIM_PORT:-5700}/api/shop_agent"
echo "Environment slots: ${SHOPSIM_ENV_SLOTS:-8}; max steps: ${SHOP_MAX_STEPS:-35}"
echo "Console log: $LOG_DIR/server.log"
exec bash "$ROOT/environments/ShopSimulator/shop_env/start.sh"
