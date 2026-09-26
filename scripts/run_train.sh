#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <prepared_dir> <model_family> <run_name>"
  exit 1
fi

PREPARED_DIR="$1"
MODEL_FAMILY="$2"
RUN_NAME="$3"

python -m src.cli train \
  --prepared-dir "${PREPARED_DIR}" \
  --model-family "${MODEL_FAMILY}" \
  --run-name "${RUN_NAME}" \
  --output-dir "reports/runs/${RUN_NAME}"
