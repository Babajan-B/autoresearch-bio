#!/usr/bin/env python3
"""Re-run full pipeline with all code fixes applied.

Fixes applied before this script:
  1. Acceptance logic: strict > comparison + revocation of superseded runs
  2. SHAP evaluation: uses validation partition (not test)
  3. SHAP background: 200 samples (up from 25)
  4. HGB class_weight: balanced (was unset)
  5. Threshold rounding: round(float(threshold), 4) to eliminate FP artifacts

This script recreates ALL runs from scratch on the frozen GSE96058 splits.
"""
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.train import run_baseline, run_training
from src.ablations import run_gene_ablation

PREPARED_DIR = "data/processed/GSE96058_ER_primary"
LOG_PATH = "results_log_v2.json"  # New log to avoid overwriting original
REPORTS_DIR = Path("reports_v2")


def run_all():
    # Clean previous v2 outputs
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
    log_file = Path(LOG_PATH)
    if log_file.exists():
        log_file.unlink()

    print("=" * 60)
    print("PHASE 1: Mandatory Logistic Regression Baseline")
    print("=" * 60)
    baseline = run_baseline(
        prepared_dir=PREPARED_DIR,
        output_dir=str(REPORTS_DIR / "baseline" / "GSE96058_ER_primary"),
        top_k_genes=1000,
        log_path=LOG_PATH,
        random_seed=42,
    )
    print(f"  Baseline: val AUROC={baseline.validation_metrics['auroc']:.4f}, "
          f"test AUROC={baseline.test_metrics['auroc']:.4f}, status={baseline.status}")

    print("\n" + "=" * 60)
    print("PHASE 2: Model Family Comparison")
    print("=" * 60)

    # Extra Trees (seed 42 = default, used for family comparison)
    et = run_training(
        prepared_dir=PREPARED_DIR,
        output_dir=str(REPORTS_DIR / "runs" / "extra_trees_gse96058_er"),
        model_family="extra_trees",
        top_k_genes=2000,
        run_name="extra_trees_gse96058_er",
        log_path=LOG_PATH,
        random_seed=42,
        notes="Paper suite model comparison on frozen GSE96058 splits.",
    )
    print(f"  Extra Trees (42): val AUROC={et.validation_metrics['auroc']:.4f}, status={et.status}")

    # Random Forest
    rf = run_training(
        prepared_dir=PREPARED_DIR,
        output_dir=str(REPORTS_DIR / "runs" / "random_forest_gse96058_er"),
        model_family="random_forest",
        top_k_genes=2000,
        run_name="random_forest_gse96058_er",
        log_path=LOG_PATH,
        random_seed=42,
        notes="Paper suite model comparison on frozen GSE96058 splits.",
    )
    print(f"  Random Forest: val AUROC={rf.validation_metrics['auroc']:.4f}, status={rf.status}")

    # Histogram Gradient Boosting (NOW WITH class_weight=balanced)
    hgb = run_training(
        prepared_dir=PREPARED_DIR,
        output_dir=str(REPORTS_DIR / "runs" / "hist_gradient_boosting_gse96058_er"),
        model_family="hist_gradient_boosting",
        top_k_genes=2000,
        run_name="hist_gradient_boosting_gse96058_er",
        log_path=LOG_PATH,
        random_seed=42,
        notes="Paper suite model comparison on frozen GSE96058 splits. Now with balanced class_weight.",
    )
    print(f"  HGB (balanced): val AUROC={hgb.validation_metrics['auroc']:.4f}, status={hgb.status}")

    print("\n" + "=" * 60)
    print("PHASE 3: Seed Stability (Extra Trees)")
    print("=" * 60)

    for seed in [7, 123]:
        s = run_training(
            prepared_dir=PREPARED_DIR,
            output_dir=str(REPORTS_DIR / "runs" / f"extra_trees_seed_{seed}"),
            model_family="extra_trees",
            top_k_genes=2000,
            run_name=f"extra_trees_seed_{seed}",
            log_path=LOG_PATH,
            random_seed=seed,
            notes="Seed stability experiment on frozen GSE96058 splits.",
        )
        print(f"  ET seed {seed}: val AUROC={s.validation_metrics['auroc']:.4f}, status={s.status}")

    print("\n" + "=" * 60)
    print("PHASE 4: Top-k Sensitivity (Extra Trees)")
    print("=" * 60)

    for k in [500, 1000, 5000]:
        tk = run_training(
            prepared_dir=PREPARED_DIR,
            output_dir=str(REPORTS_DIR / "runs" / f"extra_trees_topk_{k}"),
            model_family="extra_trees",
            top_k_genes=k,
            run_name=f"extra_trees_topk_{k}",
            log_path=LOG_PATH,
            random_seed=42,
            notes="Top-k sensitivity experiment on frozen GSE96058 splits.",
        )
        print(f"  ET top-{k}: val AUROC={tk.validation_metrics['auroc']:.4f}, status={tk.status}")

    # top-2000 is already done as extra_trees_gse96058_er above

    print("\n" + "=" * 60)
    print("PHASE 5: Shortcut-Gene Ablation")
    print("=" * 60)

    abl1 = run_gene_ablation(
        prepared_dir=PREPARED_DIR,
        output_dir=str(REPORTS_DIR / "runs" / "ablation_esr1_pgr_erbb2"),
        genes=["ESR1", "PGR", "ERBB2"],
        model_family="logistic_regression",
        top_k_genes=1000,
        log_path=LOG_PATH,
        random_seed=42,
    )
    print(f"  Ablation ESR1/PGR/ERBB2: val AUROC={abl1.validation_metrics['auroc']:.4f}")

    abl2 = run_gene_ablation(
        prepared_dir=PREPARED_DIR,
        output_dir=str(REPORTS_DIR / "runs" / "ablation_luminal_markers"),
        genes=["AGR3", "CA12", "SLC39A6"],
        model_family="logistic_regression",
        top_k_genes=1000,
        log_path=LOG_PATH,
        random_seed=42,
    )
    print(f"  Ablation AGR3/CA12/SLC39A6: val AUROC={abl2.validation_metrics['auroc']:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    # Print final summary
    log_entries = json.loads(Path(LOG_PATH).read_text())
    gse_entries = [e for e in log_entries if e.get("dataset_id") == "GSE96058"]
    accepted = [e for e in gse_entries if e.get("status") == "accepted"]
    print(f"\nTotal GSE96058 runs: {len(gse_entries)}")
    print(f"Accepted runs: {len(accepted)}")
    for a in accepted:
        print(f"  - {a['run_name']} ({a['run_type']}): val AUROC={a['validation_metrics']['auroc']:.4f}")


if __name__ == "__main__":
    run_all()
