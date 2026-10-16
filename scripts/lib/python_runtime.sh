#!/usr/bin/env bash
# Shared interpreter selection for evaluation, reports and SFT.

init_project_python() {
  local repo_root="$1"
  local runtime_label="${2:-evaluation/report}"
  PROJECT_PYTHON=""
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! PROJECT_PYTHON="$(command -v "$PYTHON_BIN")"; then
      echo "PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
      return 1
    fi
  elif PROJECT_PYTHON="$(command -v python)"; then
    :
  elif PROJECT_PYTHON="$(command -v python3)"; then
    :
  elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    PROJECT_PYTHON="$repo_root/.venv/bin/python"
  else
    echo "Cannot find Python. Activate an existing Python >=3.10 environment or set PYTHON_BIN=/path/to/python." >&2
    return 1
  fi

  # Use this checkout even when only ShopSimulator was installed. This does not
  # install packages or change the existing training/inference environment.
  export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
  printf 'Using %s Python: %s\n' "$runtime_label" "$PROJECT_PYTHON" >&2
}
