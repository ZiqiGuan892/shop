#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/setup.sh [--shopsim-only]

  --shopsim-only  Install only the isolated ShopSimulator service, product data
                  and search index. Keep the existing training environment.
  --help          Show this help without installing anything.

Without arguments, install the full SFT/GRPO stack and ShopSimulator, then apply
the repository's veRL patch. Use --shopsim-only when veRL is already installed.
Requires uv. SHOPSIM_PYTHON defaults to 3.10 (a Python executable path also works).
HELP
}

SHOPSIM_ONLY=false
case "${1:-}" in
  --shopsim-only) SHOPSIM_ONLY=true ;;
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOPSIM_ROOT="${ROOT}/environments/ShopSimulator"
SHOP_ENV_ROOT="${SHOPSIM_ROOT}/shop_env"
ENV_DIR="${SHOPSIM_ROOT}/.venv-shopsim"
COMPRESSED_PRODUCTS="${SHOP_ENV_ROOT}/data/fine_items_eval_train_all.json.gz"
PRODUCTS="${SHOP_ENV_ROOT}/data/items_eval_train.json"
EXPECTED_PRODUCT_SHA256="57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"
MAIN_PYTHON="${MAIN_PYTHON:-3.12}"
SHOPSIM_PYTHON="${SHOPSIM_PYTHON:-3.10}"

if [[ ! -f "${COMPRESSED_PRODUCTS}" ]]; then
  echo "Missing embedded product archive: ${COMPRESSED_PRODUCTS}" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the isolated ShopSimulator environment" >&2
  exit 1
fi

cd "${ROOT}"
if [[ "${SHOPSIM_ONLY}" == false ]]; then
  uv sync --python "${MAIN_PYTHON}" --extra sft --extra grpo
else
  echo "Installing ShopSimulator only; keeping the existing training environment and veRL source."
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  uv venv --python "${SHOPSIM_PYTHON}" "${ENV_DIR}"
fi
# Do not inherit the training project's NumPy override into this CPU service.
uv pip install --no-config \
  --python "${ENV_DIR}/bin/python" \
  -r "${SHOP_ENV_ROOT}/requirements.txt"
uv pip check --no-config --python "${ENV_DIR}/bin/python"

# Fail early if this Python build cannot create the SQLite FTS5 search index.
"${ENV_DIR}/bin/python" - <<'PY'
import sqlite3
import sys

with sqlite3.connect(":memory:") as connection:
    connection.execute("CREATE VIRTUAL TABLE fts5_check USING fts5(content)")
print(f"ShopSimulator Python: {sys.executable}")
print(f"SQLite {sqlite3.sqlite_version}: FTS5 available")
PY

if [[ ! -f "${PRODUCTS}" ]]; then
  temporary_products="${PRODUCTS}.preparing"
  trap 'rm -f "${temporary_products}"' EXIT
  gzip -cd "${COMPRESSED_PRODUCTS}" > "${temporary_products}"
  actual_sha256="$(sha256sum "${temporary_products}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${EXPECTED_PRODUCT_SHA256}" ]]; then
    echo "Product data SHA-256 mismatch: ${actual_sha256}" >&2
    exit 1
  fi
  mv "${temporary_products}" "${PRODUCTS}"
  trap - EXIT
fi

actual_sha256="$(sha256sum "${PRODUCTS}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_PRODUCT_SHA256}" ]]; then
  echo "Existing product data SHA-256 mismatch: ${actual_sha256}" >&2
  exit 1
fi

cd "${SHOP_ENV_ROOT}"
PYTHONPATH=. "${ENV_DIR}/bin/python" scripts/build_index.py

cd "${ROOT}"
if [[ "${SHOPSIM_ONLY}" == false ]]; then
  "${ROOT}/.venv/bin/python" scripts/apply_verl_dynamic_sampling_patch.py
  echo "Shopping Agent training environment is ready."
else
  echo "Standalone ShopSimulator environment is ready."
fi

echo "Product SHA-256: ${actual_sha256}"
echo "Index: ${SHOP_ENV_ROOT}/search_engine/products.sqlite3"
echo "Start the service: bash scripts/start_environment.sh"
