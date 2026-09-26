from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(model_family: str, random_seed: int = 42):
    family = model_family.lower()
    if family == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            random_state=random_seed,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            class_weight="balanced",
            random_state=random_seed,
        )
    raise ValueError(f"Unsupported model family: {model_family}")


def predict_probabilities(model, features):
    probabilities = model.predict_proba(features)
    if probabilities.shape[1] == 2:
        return probabilities
    return probabilities


def save_model(model, output_path: str | Path) -> Path:
    target = Path(output_path)
    with target.open("wb") as handle:
        pickle.dump(model, handle)
    return target


def load_model(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def model_family_name(model) -> str:
    return model.named_steps["model"].__class__.__name__ if isinstance(model, Pipeline) else model.__class__.__name__


def positive_class_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.shape[1] == 2:
        return probabilities[:, 1]
    raise ValueError("Positive-class probabilities are only defined for binary classifiers")
