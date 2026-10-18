from __future__ import annotations

from pathlib import Path

from src.logging_utils import read_results_log
from src.train import TrainingRunSummary, run_training
from src.utils.io import ensure_dir


def run_gene_ablation(
    *,
    prepared_dir: str | Path,
    output_dir: str | Path,
    genes: list[str],
    model_family: str = "logistic_regression",
    top_k_genes: int = 1000,
    log_path: str | Path = "results_log.json",
    random_seed: int = 42,
) -> TrainingRunSummary:
    summary = run_training(
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        model_family=model_family,
        top_k_genes=top_k_genes,
        excluded_genes=genes,
        log_path=log_path,
        random_seed=random_seed,
        notes=f"Shortcut-gene ablation excluding: {', '.join(genes)}",
        run_type="ablation",
        run_name=f"ablation_{model_family}",
    )
    _write_ablation_report(output_dir=output_dir, dataset_id=summary.dataset_id, genes=genes, log_path=log_path)
    return summary


def _write_ablation_report(
    *,
    output_dir: str | Path,
    dataset_id: str,
    genes: list[str],
    log_path: str | Path,
) -> None:
    output_path = ensure_dir(output_dir)
    entries = [entry for entry in read_results_log(log_path) if entry.get("dataset_id") == dataset_id]
    ablation_entries = [entry for entry in entries if entry.get("run_type") == "ablation"]
    baseline_entries = [entry for entry in entries if entry.get("run_type") == "baseline"]
    latest_ablation = ablation_entries[-1] if ablation_entries else None
    baseline = baseline_entries[-1] if baseline_entries else None

    lines = ["# Ablation Report", "", f"- Dataset: {dataset_id}", f"- Excluded genes: {', '.join(genes)}", ""]
    if baseline:
        lines.append(
            "- Baseline validation AUROC: "
            f"{baseline['validation_metrics']['auroc']:.4f}, "
            f"Macro-F1: {baseline['validation_metrics']['macro_f1']:.4f}"
        )
    if latest_ablation:
        lines.append(
            "- Ablation validation AUROC: "
            f"{latest_ablation['validation_metrics']['auroc']:.4f}, "
            f"Macro-F1: {latest_ablation['validation_metrics']['macro_f1']:.4f}"
        )
    lines.append("")
    (output_path / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
