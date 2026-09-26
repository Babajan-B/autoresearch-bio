from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.shap_utils import _sample_frame, export_shap_rankings


def test_sample_frame_preserves_minority_class_when_labels_provided() -> None:
    frame = pd.DataFrame(
        {"g1": np.arange(100), "g2": np.arange(100, 200)},
        index=[f"s{i}" for i in range(100)],
    )
    labels = pd.Series(
        ["majority"] * 95 + ["minority"] * 5,
        index=frame.index,
    )

    sampled = _sample_frame(frame, 40, labels, random_state=42)

    sampled_labels = labels.loc[sampled.index]
    assert len(sampled) == 40
    assert "minority" in sampled_labels.unique()


def test_export_shap_rankings_uses_sampled_background_for_linear_explainer(
    tmp_path,
    monkeypatch,
) -> None:
    rng = np.random.default_rng(42)
    background = pd.DataFrame(
        rng.normal(size=(120, 4)),
        columns=["g1", "g2", "g3", "g4"],
        index=[f"train_{i}" for i in range(120)],
    )
    evaluation = pd.DataFrame(
        rng.normal(size=(30, 4)),
        columns=background.columns,
        index=[f"val_{i}" for i in range(30)],
    )
    y = np.array([0] * 96 + [1] * 24)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(background, y)

    captured: dict[str, tuple[int, int]] = {}

    class FakeLinearExplainer:
        def __init__(self, estimator, background_matrix):
            captured["background_shape"] = background_matrix.shape

        def __call__(self, evaluation_matrix):
            captured["evaluation_shape"] = evaluation_matrix.shape
            return SimpleNamespace(values=np.zeros((evaluation_matrix.shape[0], evaluation_matrix.shape[1])))

    monkeypatch.setattr("src.shap_utils.shap.LinearExplainer", FakeLinearExplainer)
    monkeypatch.setattr("src.shap_utils.save_shap_bar_plot", lambda genes, values, path: None)

    ranking, csv_path = export_shap_rankings(
        model=model,
        background=background,
        evaluation=evaluation,
        output_dir=tmp_path,
        top_n=4,
        background_sample_size=50,
        background_labels=pd.Series(y, index=background.index),
    )

    assert captured["background_shape"] == (50, 4)
    assert captured["evaluation_shape"] == (30, 4)
    assert len(ranking) == 4
    assert csv_path.exists()
