#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-reports/final/final_report.md}"
python -m src.cli report --output-path "${OUTPUT_PATH}"
