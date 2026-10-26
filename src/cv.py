from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.features import select_top_variable_genes
from src.metrics import probability_metrics
from src.models import build_model, positive_class_probabilities


def cross_validated_metrics(
    expression: pd.DataFrame,
    y: np.ndarray,
    model_family: str,
    top_k_genes: int,
    excluded_genes: list[str] | None = None,
    folds: int = 5,
    random_seed: int = 42,
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and standard error over stratified folds; genes are re-selected inside every training fold."""
    per_fold = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    for train_idx, test_idx in splitter.split(expression, y):
        genes = select_top_variable_genes(expression.iloc[train_idx], top_k_genes, excluded_genes).selected_genes
        model = build_model(model_family, random_seed=random_seed)
        model.fit(expression.iloc[train_idx][genes], y[train_idx])
        p = positive_class_probabilities(model.predict_proba(expression.iloc[test_idx][genes]))
        per_fold.append(probability_metrics(y[test_idx], p))
    frame = pd.DataFrame(per_fold)
    return frame.mean().to_dict(), (frame.std(ddof=1) / np.sqrt(len(frame))).to_dict()
