#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports/runs
LOG_FILE="reports/runs/gse96058_default_benchmark.log"
: > "${LOG_FILE}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "[runner] starting default benchmark at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python -m src.cli prepare \
  --benchmark-preset gse96058_er \
  --input-mode geo \
  --label-column "er status" \
  --output-dir data/processed/GSE96058_ER_primary

echo "[runner] prepare completed, starting baseline"

python -m src.cli baseline \
  --prepared-dir data/processed/GSE96058_ER_primary \
  --output-dir reports/baseline/GSE96058_ER_primary

echo "[runner] baseline completed"

