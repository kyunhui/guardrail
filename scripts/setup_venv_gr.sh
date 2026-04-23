#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/venv_gr"
REQ_FILE="${ROOT_DIR}/requirements.txt"

echo "=============================================="
echo " Guardrail venv setup (venv_gr)"
echo "=============================================="
echo "Project root : ${ROOT_DIR}"
echo "Venv path    : ${VENV_DIR}"
echo "Requirements : ${REQ_FILE}"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found. Please install Python 3.10+ first."
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[ERROR] requirements.txt not found at: ${REQ_FILE}"
  exit 1
fi

if [[ -d "${VENV_DIR}" ]]; then
  echo "[INFO] Existing venv_gr found. Reusing it."
else
  echo "[INFO] Creating virtual environment: venv_gr"
  python3 -m venv "${VENV_DIR}"
fi

echo "[INFO] Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[INFO] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing dependencies from requirements.txt..."
python -m pip install -r "${REQ_FILE}"

echo
echo "[DONE] venv_gr setup complete."
echo "To activate manually:"
echo "  source \"${VENV_DIR}/bin/activate\""
echo
echo "Then run evaluation with:"
echo "  bash scripts/run_eval.sh --model_name WildGuard --tasks toxicchat --report_output_path ./classification_results/metrics.json"
