from pathlib import Path

from src.logging_utils import append_result, baseline_exists, read_results_log, render_markdown_report


def test_logging_appends_and_renders_report(tmp_path: Path) -> None:
    log_path = tmp_path / "results_log.json"
    log_path.write_text("[]", encoding="utf-8")

    entry = {
        "dataset_id": "local_smoke",
        "run_name": "baseline_logistic_regression",
        "run_type": "baseline",
        "model_family": "logistic_regression",
        "status": "accepted",
        "validation_metrics": {"auroc": 0.8, "macro_f1": 0.7, "brier_score": 0.2, "mcc": 0.6},
        "test_metrics": {"auroc": 0.79, "macro_f1": 0.69, "brier_score": 0.21, "mcc": 0.58},
        "output_dir": "reports/baseline/local_smoke",
    }
    append_result(entry, log_path)

    entries = read_results_log(log_path)
    assert baseline_exists(log_path, "local_smoke")
    assert len(entries) == 1
    assert "Best run" in render_markdown_report(entries)
