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


def calibration_slope(y_true: np.ndarray, positive_probabilities: np.ndarray) -> float:
    """Slope of an unpenalised logistic recalibration of y on logit(p); 1 is ideal, <1 too extreme, >1 too conservative."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(positive_probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    # C=1e10 is effectively unpenalised and avoids penalty=None, deprecated in scikit-learn 1.8
    return float(LogisticRegression(C=1e10, max_iter=1000).fit(logit, y_true).coef_[0, 0])


def probability_metrics(y_true: np.ndarray, positive_probabilities: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics used by the promotion policies and the selection study."""
    from sklearn.metrics import average_precision_score

    slope = calibration_slope(y_true, positive_probabilities)
    minority = int(np.mean(y_true) < 0.5)
    minority_scores = positive_probabilities if minority == 1 else 1 - positive_probabilities
    return {
        "auroc": float(roc_auc_score(y_true, positive_probabilities)),
        # minority class as positive, so precision-recall reflects the rarer phenotype
        "average_precision": float(average_precision_score(y_true == minority, minority_scores)),
        "brier_score": float(np.mean((positive_probabilities - y_true) ** 2)),
        "calibration_slope": slope,
        "abs_log_calibration_slope": float(abs(np.log(slope))) if slope > 0 else float("inf"),
    }

