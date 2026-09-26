from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import read_json, write_json


def read_results_log(log_path: str | Path = "results_log.json") -> list[dict[str, Any]]:
    payload = read_json(log_path, default=[])
    if not isinstance(payload, list):
        raise ValueError("results_log.json must contain a JSON array")
    return payload


def metric_rank_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("auroc", 0.0)),
        float(metrics.get("macro_f1", 0.0)),
        -float(metrics.get("brier_score", 1.0)),
        float(metrics.get("mcc", 0.0)),
    )


def baseline_exists(log_path: str | Path, dataset_id: str) -> bool:
    return any(
        entry.get("dataset_id") == dataset_id and entry.get("run_type") == "baseline"
        for entry in read_results_log(log_path)
    )


def determine_acceptance(
    log_path: str | Path,
    dataset_id: str,
    validation_metrics: dict[str, Any],
    run_type: str,
) -> str:
    if run_type == "baseline":
        return "accepted"

    previous_entries = [
        entry
        for entry in read_results_log(log_path)
        if entry.get("dataset_id") == dataset_id and entry.get("status") == "accepted"
    ]
    if not previous_entries:
        return "accepted"

    previous_best = max(previous_entries, key=lambda entry: metric_rank_tuple(entry["validation_metrics"]))
    if metric_rank_tuple(validation_metrics) > metric_rank_tuple(previous_best["validation_metrics"]):
        return "accepted"
    return "rejected"


def _revoke_previous_accepted(log_path: str | Path, dataset_id: str, keep_run_name: str) -> None:
    """Revoke acceptance from all non-baseline runs except the designated best."""
    log_entries = read_results_log(log_path)
    changed = False
    for entry in log_entries:
        if (
            entry.get("dataset_id") == dataset_id
            and entry.get("status") == "accepted"
            and entry.get("run_type") != "baseline"
            and entry.get("run_name") != keep_run_name
        ):
            entry["status"] = "superseded"
            changed = True
    if changed:
        write_json(log_path, log_entries)


def append_result(entry: dict[str, Any], log_path: str | Path = "results_log.json") -> None:
    log_entries = read_results_log(log_path)
    log_entries.append(entry)
    write_json(log_path, log_entries)

    if entry.get("status") == "accepted" and entry.get("run_type") != "baseline":
        _revoke_previous_accepted(
            log_path,
            entry["dataset_id"],
            keep_run_name=entry["run_name"],
        )


def render_markdown_report(log_entries: list[dict[str, Any]]) -> str:
    if not log_entries:
        return "# Autoresearch-Bio Report\n\nNo runs have been logged yet.\n"

    lines = ["# Autoresearch-Bio Report", ""]
    datasets = sorted({entry.get("dataset_id", "unknown") for entry in log_entries})
    for dataset_id in datasets:
        dataset_entries = [entry for entry in log_entries if entry.get("dataset_id") == dataset_id]
        accepted = [entry for entry in dataset_entries if entry.get("status") == "accepted"]
        best_entry = max(accepted or dataset_entries, key=lambda entry: metric_rank_tuple(entry["validation_metrics"]))

        lines.append(f"## {dataset_id}")
        lines.append("")
        lines.append(f"- Total runs: {len(dataset_entries)}")
        lines.append(f"- Accepted runs: {sum(entry.get('status') == 'accepted' for entry in dataset_entries)}")
        lines.append(f"- Best run: {best_entry.get('run_name')}")
        lines.append(f"- Best model family: {best_entry.get('model_family')}")
        lines.append(
            "- Validation metrics: "
            f"AUROC={best_entry['validation_metrics']['auroc']:.4f}, "
            f"Macro-F1={best_entry['validation_metrics']['macro_f1']:.4f}, "
            f"Brier={best_entry['validation_metrics']['brier_score']:.4f}, "
            f"MCC={best_entry['validation_metrics']['mcc']:.4f}"
        )
        lines.append(
            "- Test metrics: "
            f"AUROC={best_entry['test_metrics']['auroc']:.4f}, "
            f"Macro-F1={best_entry['test_metrics']['macro_f1']:.4f}, "
            f"Brier={best_entry['test_metrics']['brier_score']:.4f}, "
            f"MCC={best_entry['test_metrics']['mcc']:.4f}"
        )
        lines.append(f"- Output directory: {best_entry.get('output_dir')}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
