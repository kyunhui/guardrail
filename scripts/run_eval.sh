#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/venv_gr"
EVAL_PY="${ROOT_DIR}/evaluation/eval.py"

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_eval.sh --model_name <MODEL_NAME> --tasks <TASKS> --report_output_path <REPORT_PATH> [--save_individual_results_path <ALL_PATH>] [--override_model_path <MODEL_PATH>] [--override_existing_report true|false]

Required arguments:
  --model_name                  Classifier model name (e.g., WildGuard, LlamaGuard2, Qwen25VLInstruct)
  --tasks                       Comma-separated tasks (e.g., toxicchat,openai_mod,xstest_response_refusal)
  --report_output_path          Output JSON path for aggregate metrics

Optional arguments:
  --save_individual_results_path  Output JSON path for per-sample predictions
  --override_model_path           Local model path override
  --override_existing_report      true/false (default false)

Examples:
  bash scripts/run_eval.sh \
    --model_name WildGuard \
    --tasks wildguardtest_prompt,wildguardtest_response,wildguardtest_refusal \
    --report_output_path ./classification_results/wildguard_metrics.json \
    --save_individual_results_path ./classification_results/wildguard_all.json

  bash scripts/run_eval.sh \
    --model_name Qwen25VLInstruct \
    --tasks toxicchat,openai_mod,simplesafetytests,harmbench,wildguardtest_prompt,harmimage,spa_vl,harmbench:response,saferlhf,beavertails,xstest_response_harm,wildguardtest_response,spa_vl:response,xstest_response_refusal \
    --report_output_path ./classification_results/qwen25vl_metrics.json \
    --save_individual_results_path ./classification_results/qwen25vl_all.json
EOF
}

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[ERROR] venv_gr not found at ${VENV_DIR}"
  echo "Please run setup first:"
  echo "  bash scripts/setup_venv_gr.sh"
  exit 1
fi

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "[ERROR] evaluation entrypoint not found: ${EVAL_PY}"
  exit 1
fi

if [[ $# -eq 0 ]]; then
  print_usage
  exit 1
fi

echo "=============================================="
echo " Safety Classifier Evaluation Runner"
echo "=============================================="
echo "Project root : ${ROOT_DIR}"
echo "Using venv   : ${VENV_DIR}"
echo

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[INFO] Running classification evaluation..."
echo "[INFO] Command: python evaluation/eval.py classifiers $*"
echo

cd "${ROOT_DIR}"
python "${EVAL_PY}" classifiers "$@"

echo
echo "[DONE] Evaluation finished."
