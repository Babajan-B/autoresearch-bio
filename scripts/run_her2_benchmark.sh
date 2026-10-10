#!/usr/bin/env bash
# HER2 benchmark through the locked CLI. Decisions were declared in config/her2_declaration.md before any HER2 model was trained.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
P=data/processed/GSE96058_HER2_primary
LOG=reports/her2/results_log_her2.json
R=reports/her2/runs
mkdir -p "$R"
[ -f "$P/splits.json" ] || $PY -m src.cli prepare --benchmark-preset gse96058_her2 --input-mode geo --label-column "her2 status" --output-dir "$P"
$PY -m src.cli baseline --prepared-dir "$P" --output-dir "$R/baseline" --log-path "$LOG"
t() { $PY -m src.cli train --prepared-dir "$P" --model-family "$1" --run-name "$2" --output-dir "$R/$2" --top-k-genes "$3" --random-seed "$4" --log-path "$LOG" --notes "HER2 benchmark"; }
t random_forest her2_random_forest 2000 42
t hist_gradient_boosting her2_hist_gradient_boosting 2000 42
for s in 7 42 123; do t extra_trees her2_extra_trees_seed_$s 2000 $s; done
for k in 500 1000 5000; do t extra_trees her2_extra_trees_topk_$k $k 42; done
$PY -m src.cli ablate --prepared-dir "$P" --genes ERBB2,GRB7,STARD3 --output-dir "$R/ablation_17q12" --random-seed 42 --log-path "$LOG"
echo HER2_SUITE_DONE
