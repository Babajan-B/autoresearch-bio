#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations import run_gene_ablation
from src.train import run_baseline
from src.utils.io import load_prepared_dataset


PREPARED_DIR = Path("data/processed/GSE96058_ER_primary")
TEMP_LOG = Path("artifacts/temp_core_refresh_log.json")


def main() -> None:
    dataset_id = load_prepared_dataset(PREPARED_DIR).bundle.dataset_id
    TEMP_LOG.parent.mkdir(parents=True, exist_ok=True)
    TEMP_LOG.write_text("[]\n", encoding="utf-8")

    run_baseline(
        prepared_dir=PREPARED_DIR,
        output_dir=Path("reports/baseline/GSE96058_ER_primary"),
        top_k_genes=1000,
        log_path=TEMP_LOG,
        random_seed=42,
    )

    run_gene_ablation(
        prepared_dir=PREPARED_DIR,
        output_dir=Path("reports/runs/ablation_esr1_pgr_erbb2"),
        genes=["ESR1", "PGR", "ERBB2"],
        model_family="logistic_regression",
        top_k_genes=1000,
        log_path=TEMP_LOG,
        random_seed=42,
    )

    run_gene_ablation(
        prepared_dir=PREPARED_DIR,
        output_dir=Path("reports/runs/ablation_luminal_markers"),
        genes=["AGR3", "CA12", "SLC39A6"],
        model_family="logistic_regression",
        top_k_genes=1000,
        log_path=TEMP_LOG,
        random_seed=42,
    )

    print(json.dumps({"dataset_id": dataset_id, "status": "refreshed"}, indent=2))


if __name__ == "__main__":
    main()
