from pathlib import Path

import numpy as np

from src.evaluate import evaluate_predictions


def test_evaluate_writes_metrics_and_plots(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )
    result = evaluate_predictions(
        y_true=y_true,
        probabilities=probabilities,
        labels=["negative", "positive"],
        output_dir=tmp_path,
        split_name="test",
        sample_ids=["S1", "S2", "S3", "S4"],
        threshold=0.5,
    )
    assert result.metrics["auroc"] == 1.0
    assert (tmp_path / "test_roc.png").exists()
    assert (tmp_path / "test_confusion.png").exists()
    assert (tmp_path / "test_calibration.png").exists()
