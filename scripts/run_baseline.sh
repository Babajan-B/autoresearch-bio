#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <expression_path> <phenotype_path> <label_column> <dataset_name>"
  exit 1
fi

EXPR_PATH="$1"
PHENO_PATH="$2"
LABEL_COLUMN="$3"
DATASET_NAME="$4"

python -m src.cli prepare \
  --input-mode local \
  --expression-path "${EXPR_PATH}" \
  --phenotype-path "${PHENO_PATH}" \
  --label-column "${LABEL_COLUMN}" \
  --output-dir "data/processed/${DATASET_NAME}"

python -m src.cli baseline \
  --prepared-dir "data/processed/${DATASET_NAME}" \
  --output-dir "reports/baseline/${DATASET_NAME}"
