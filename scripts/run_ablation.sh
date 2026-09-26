#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <prepared_dir> [comma_separated_genes]"
  exit 1
fi

PREPARED_DIR="$1"
GENES="${2:-ESR1,PGR,ERBB2}"

python -m src.cli ablate \
  --prepared-dir "${PREPARED_DIR}" \
  --genes "${GENES}" \
  --output-dir "reports/runs/ablation_$(date +%Y%m%d_%H%M%S)"
