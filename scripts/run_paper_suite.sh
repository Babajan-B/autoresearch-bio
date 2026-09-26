#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PREPARED_DIR="${1:-data/processed/GSE96058_ER_primary}"
LOG_FILE="${2:-reports/runs/gse96058_paper_suite.log}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

mkdir -p reports/runs reports/final
: > "${LOG_FILE}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "[paper-suite] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[paper-suite] prepared_dir=${PREPARED_DIR}"

run_train() {
  local model_family="$1"
  local run_name="$2"
  local output_dir="$3"
  local top_k="$4"
  local random_seed="$5"
  local notes="$6"

  echo "[paper-suite] train model=${model_family} run_name=${run_name} top_k=${top_k} seed=${random_seed}"
  "${PYTHON_BIN}" -m src.cli train \
    --prepared-dir "${PREPARED_DIR}" \
    --model-family "${model_family}" \
    --run-name "${run_name}" \
    --output-dir "${output_dir}" \
    --top-k-genes "${top_k}" \
    --random-seed "${random_seed}" \
    --notes "${notes}"
}

run_ablation() {
  local genes="$1"
  local output_dir="$2"

  echo "[paper-suite] ablation genes=${genes}"
  "${PYTHON_BIN}" -m src.cli ablate \
    --prepared-dir "${PREPARED_DIR}" \
    --genes "${genes}" \
    --output-dir "${output_dir}" \
    --random-seed 42
}

run_train "random_forest" "random_forest_gse96058_er" "reports/runs/random_forest_gse96058_er" 2000 42 \
  "Paper suite model comparison on frozen GSE96058 splits."
run_train "hist_gradient_boosting" "hist_gradient_boosting_gse96058_er" "reports/runs/hist_gradient_boosting_gse96058_er" 2000 42 \
  "Paper suite model comparison on frozen GSE96058 splits."

for seed in 7 42 123; do
  run_train "extra_trees" "extra_trees_seed_${seed}" "reports/runs/extra_trees_seed_${seed}" 2000 "${seed}" \
    "Seed stability experiment on frozen GSE96058 splits."
done

for top_k in 500 1000 2000 5000; do
  run_train "extra_trees" "extra_trees_topk_${top_k}" "reports/runs/extra_trees_topk_${top_k}" "${top_k}" 42 \
    "Top-k sensitivity experiment on frozen GSE96058 splits."
done

run_ablation "AGR3,CA12,SLC39A6" "reports/runs/ablation_luminal_markers"

echo "[paper-suite] exporting summary report"
"${PYTHON_BIN}" -m src.cli report --output-path reports/final/gse96058_paper_suite_report.md

echo "[paper-suite] completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
