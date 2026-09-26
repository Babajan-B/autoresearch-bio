#!/usr/bin/env bash
# Recompute every statistic, table and figure reported in the manuscript
# from the per-run artefacts written by the pipeline (reports/ and data/processed/).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:-$(dirname "$HERE")}"
export PAPER_OUT="${PAPER_OUT:-$AUTORESEARCH_ROOT/reports/paper}"
cd "$HERE"
python analysis.py      # 95% CIs (DeLong, bootstrap), paired DeLong tests, ablation deltas
python external.py      # GSE81538 metrics vs PAM50 proxy and ER pathology; calibration slopes
python table1.py        # cohort characteristics by partition (Table 1)
python mfig1.py         # Fig. 1  cohort flow
python mfigs.py         # Figs. 2-4 and Supplementary Figs. S2-S3
python sfig1.py         # Supplementary Fig. S1 framework architecture
echo "Outputs written to $PAPER_OUT"
