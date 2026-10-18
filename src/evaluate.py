from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import compute_classification_metrics, probabilities_to_predictions
from src.utils.io import ensure_dir, write_json
from src.utils.plots import save_calibration_plot, save_confusion_matrix, save_roc_curve


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    plot_paths: dict[str, str]
    predictions_path: str


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    output_dir: str | Path,
    split_name: str,
    sample_ids: list[str],
    threshold: float | None = None,
) -> EvaluationResult:
    output_path = ensure_dir(output_dir)
    metrics = compute_classification_metrics(y_true, probabilities, labels=labels, threshold=threshold)
    predictions = probabilities_to_predictions(probabilities, threshold=threshold)

    predictions_frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "y_true": y_true,
            "y_pred": predictions,
        }
    )
    if probabilities.ndim == 1:
        predictions_frame["probability"] = probabilities
    else:
        for class_index in range(probabilities.shape[1]):
            predictions_frame[f"probability_class_{class_index}"] = probabilities[:, class_index]

    predictions_path = output_path / f"{split_name}_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)

    roc_path = save_roc_curve(y_true, probabilities, output_path / f"{split_name}_roc.png")
    confusion_path = save_confusion_matrix(y_true, predictions, labels, output_path / f"{split_name}_confusion.png")
    calibration_path = save_calibration_plot(y_true, probabilities, output_path / f"{split_name}_calibration.png")

    plot_paths = {
        "roc_curve": str(roc_path),
        "confusion_matrix": str(confusion_path),
        "calibration_plot": str(calibration_path),
    }
    write_json(output_path / f"{split_name}_metrics.json", metrics)
    return EvaluationResult(metrics=metrics, plot_paths=plot_paths, predictions_path=str(predictions_path))
