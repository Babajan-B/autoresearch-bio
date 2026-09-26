from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def choose_binary_threshold(y_true: np.ndarray, positive_probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.1, 0.9, 17):
        predictions = (positive_probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, average="macro")
        if score > best_score:
            best_score = score
            best_threshold = round(float(threshold), 4)
    return best_threshold


def probabilities_to_predictions(probabilities: np.ndarray, threshold: float | None = None) -> np.ndarray:
    if probabilities.ndim == 1:
        if threshold is None:
            threshold = 0.5
        return (probabilities >= threshold).astype(int)
    if probabilities.shape[1] == 2 and threshold is not None:
        return (probabilities[:, 1] >= threshold).astype(int)
    return np.argmax(probabilities, axis=1)


def compute_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if probabilities.ndim == 1:
        positive_prob = probabilities
        return float(np.mean((positive_prob - y_true) ** 2))

    if probabilities.shape[1] == 2:
        return float(np.mean((probabilities[:, 1] - y_true) ** 2))

    classes = probabilities.shape[1]
    one_hot = np.eye(classes)[y_true]
    return float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1)))


def compute_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    threshold: float | None = None,
) -> dict[str, Any]:
    predictions = probabilities_to_predictions(probabilities, threshold=threshold)

    if probabilities.ndim == 1:
        auroc = roc_auc_score(y_true, probabilities)
    elif probabilities.shape[1] == 2:
        auroc = roc_auc_score(y_true, probabilities[:, 1])
    else:
        auroc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")

    metrics = {
        "auroc": float(auroc),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "brier_score": compute_brier_score(y_true, probabilities),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "labels": labels,
    }
    if threshold is not None:
        metrics["threshold"] = float(threshold)
    return metrics
