from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.utils.io import ensure_dir
from src.utils.plots import save_shap_bar_plot


def _extract_feature_matrices(model, background: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[object, np.ndarray, np.ndarray]:
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
        if len(model.steps) > 1:
            transformer = Pipeline(model.steps[:-1])
            return estimator, transformer.transform(background), transformer.transform(evaluation)
        return estimator, background.to_numpy(), evaluation.to_numpy()
    return model, background.to_numpy(), evaluation.to_numpy()


def _normalize_shap_values(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        if values.shape[2] == 2:
            return values[:, :, 1]
        return np.mean(np.abs(values), axis=2)
    return values


def _sample_frame(
    frame: pd.DataFrame,
    sample_size: int,
    labels: pd.Series | None = None,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(frame) <= sample_size:
        return frame.copy()

    if labels is None:
        return frame.sample(sample_size, random_state=random_state)

    aligned = labels.loc[frame.index].astype(str)
    class_counts = aligned.value_counts()
    if class_counts.empty:
        return frame.sample(sample_size, random_state=random_state)

    target = min(sample_size, len(frame))
    expected = class_counts / len(frame) * target
    allocation = np.floor(expected).astype(int)

    if len(class_counts) <= target:
        allocation = allocation.clip(lower=1)
    allocation = allocation.clip(upper=class_counts)

    total = int(allocation.sum())
    remainder = expected - allocation

    if total > target:
        for label in remainder.sort_values().index:
            while total > target and allocation[label] > 1:
                allocation[label] -= 1
                total -= 1
            if total == target:
                break

    if total < target:
        for label in remainder.sort_values(ascending=False).index:
            available = int(class_counts[label] - allocation[label])
            if available <= 0:
                continue
            add = min(available, target - total)
            allocation[label] += add
            total += add
            if total == target:
                break

    sampled_groups: list[pd.DataFrame] = []
    for offset, label in enumerate(class_counts.index):
        take = int(allocation[label])
        if take <= 0:
            continue
        subset = frame.loc[aligned[aligned == label].index]
        sampled_groups.append(subset.sample(take, random_state=random_state + offset))

    sampled = pd.concat(sampled_groups, axis=0)
    return sampled.sample(frac=1.0, random_state=random_state)


def export_shap_rankings(
    model,
    background: pd.DataFrame,
    evaluation: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 50,
    background_sample_size: int = 200,
    evaluation_sample_size: int = 10,
    background_labels: pd.Series | None = None,
    evaluation_labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, Path]:
    output_path = ensure_dir(output_dir)
    background_sample = _sample_frame(
        background,
        background_sample_size,
        background_labels,
        random_state=42,
    )
    evaluation_sample = _sample_frame(
        evaluation,
        evaluation_sample_size,
        evaluation_labels,
        random_state=42,
    )
    estimator, background_matrix, evaluation_matrix = _extract_feature_matrices(model, background_sample, evaluation)

    importance_by_gene = pd.Series(0.0, index=evaluation.columns, dtype=float)

    try:
        if hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, background_matrix)
            explanation = explainer(evaluation_matrix)
            raw_values = explanation.values if hasattr(explanation, "values") else np.asarray(explanation)
            normalized = _normalize_shap_values(np.asarray(raw_values))
            if normalized.ndim != 2:
                raise ValueError("Unexpected SHAP output shape")
            importance_by_gene.loc[evaluation.columns] = np.mean(np.abs(normalized), axis=0)
        elif hasattr(estimator, "estimators_"):
            explainer = shap.TreeExplainer(estimator)
            explanation = explainer(evaluation_matrix)
            raw_values = explanation.values if hasattr(explanation, "values") else np.asarray(explanation)
            normalized = _normalize_shap_values(np.asarray(raw_values))
            if normalized.ndim != 2:
                raise ValueError("Unexpected SHAP output shape")
            importance_by_gene.loc[evaluation.columns] = np.mean(np.abs(normalized), axis=0)
        else:
            evaluation_small = evaluation_sample.head(1)
            permutation_max_evals = max(2 * evaluation_small.shape[1] + 1, 250)
            explainer = shap.Explainer(model.predict_proba, background_sample, algorithm="permutation")
            explanation = explainer(
                evaluation_small,
                max_evals=permutation_max_evals,
                silent=True,
            )
            raw_values = explanation.values if hasattr(explanation, "values") else np.asarray(explanation)
            normalized = _normalize_shap_values(np.asarray(raw_values))
            if normalized.ndim != 2:
                raise ValueError("Unexpected SHAP output shape")
            importance_by_gene.loc[evaluation_small.columns] = np.mean(np.abs(normalized), axis=0)
    except Exception:
        evaluation_small = evaluation_sample.head(1)
        permutation_max_evals = max(2 * evaluation_small.shape[1] + 1, 250)
        explainer = shap.Explainer(model.predict_proba, background_sample, algorithm="permutation")
        explanation = explainer(
            evaluation_small,
            max_evals=permutation_max_evals,
            silent=True,
        )
        raw_values = explanation.values if hasattr(explanation, "values") else np.asarray(explanation)
        normalized = _normalize_shap_values(np.asarray(raw_values))
        if normalized.ndim != 2:
            raise ValueError("Unexpected SHAP output shape")
        importance_by_gene.loc[evaluation_small.columns] = np.mean(np.abs(normalized), axis=0)

    ranking = (
        pd.DataFrame({"gene": evaluation.columns, "mean_abs_shap": importance_by_gene.to_numpy()})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    csv_path = output_path / "top_genes_by_shap.csv"
    ranking.to_csv(csv_path, index=False)
    save_shap_bar_plot(ranking["gene"].tolist(), ranking["mean_abs_shap"].to_numpy(), output_path / "top_genes_by_shap.png")
    return ranking, csv_path
