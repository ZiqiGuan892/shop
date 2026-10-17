#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/lib/python_runtime.sh"
init_project_python "$ROOT" GRPO
export PYTHONUNBUFFERED=1
exec "$PROJECT_PYTHON" scripts/train_grpo.py "$@"
