from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.evaluate import evaluate_predictions
from src.features import select_top_variable_genes
from src.logging_utils import append_result, baseline_exists, determine_acceptance
from src.metrics import choose_binary_threshold, probability_metrics
from src.models import build_model, positive_class_probabilities, predict_probabilities, save_model
from src.shap_utils import export_shap_rankings
from src.cv import cross_validated_metrics
from src.promotion import load_policy
from src.provenance import collect_provenance
from src.utils.io import ensure_dir, load_prepared_dataset, sha256_file, utc_now


@dataclass(slots=True)
class TrainingRunSummary:
    run_name: str
    run_type: str
    model_family: str
    dataset_id: str
    output_dir: str
    feature_count: int
    status: str
    runtime_seconds: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


def _encode_labels(phenotype: pd.DataFrame, label_column: str, sample_ids: list[str]) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    values = phenotype.loc[sample_ids, label_column].astype(str)
    return encoder.fit_transform(values), encoder


def _slice_expression(expression: pd.DataFrame, sample_ids: list[str], genes: list[str]) -> pd.DataFrame:
    return expression.loc[sample_ids, genes]


def run_training(
    *,
    prepared_dir: str | Path,
    output_dir: str | Path,
    model_family: str,
    top_k_genes: int = 2000,
    run_name: str | None = None,
    excluded_genes: list[str] | None = None,
    baseline: bool = False,
    log_path: str | Path = "results_log.json",
    random_seed: int = 42,
    notes: str = "",
    run_type: str = "train",
    declaration_path: str | Path | None = None,
    require_clean: bool = False,
    policy_path: str | Path | None = None,
    cv_folds: int = 5,
) -> TrainingRunSummary:
    policy = load_policy(policy_path) if policy_path else None
    provenance = collect_provenance(prepared_dir, declaration_path, require_clean)
    prepared = load_prepared_dataset(prepared_dir)
    dataset_id = prepared.bundle.dataset_id
    if not baseline and not baseline_exists(log_path, dataset_id):
        raise RuntimeError(f"A baseline must be logged for {dataset_id} before running non-baseline training")

    output_path = ensure_dir(output_dir)
    run_label = run_name or f"{run_type}_{model_family}"
    start = perf_counter()

    expression = prepared.bundle.expression
    phenotype = prepared.bundle.phenotype
    label_column = prepared.bundle.label_column
    splits = prepared.splits

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    y_train, encoder = _encode_labels(phenotype, label_column, train_ids)
    y_val = encoder.transform(phenotype.loc[val_ids, label_column].astype(str))
    y_test = encoder.transform(phenotype.loc[test_ids, label_column].astype(str))

    feature_result = select_top_variable_genes(
        expression.loc[train_ids],
        top_k_genes=top_k_genes,
        excluded_genes=excluded_genes,
    )
    selected_genes = feature_result.selected_genes

    x_train = _slice_expression(expression, train_ids, selected_genes)
    x_val = _slice_expression(expression, val_ids, selected_genes)
    x_test = _slice_expression(expression, test_ids, selected_genes)

    model = build_model(model_family, random_seed=random_seed)
    model.fit(x_train, y_train)

    val_probabilities = predict_probabilities(model, x_val)
    threshold = None
    if len(encoder.classes_) == 2:
        threshold = choose_binary_threshold(y_val, positive_class_probabilities(val_probabilities))

    evaluation_dir = output_path / "evaluation"
    validation_result = evaluate_predictions(
        y_true=y_val,
        probabilities=val_probabilities,
        labels=encoder.classes_.astype(str).tolist(),
        output_dir=evaluation_dir,
        split_name="validation",
        sample_ids=val_ids,
        threshold=threshold,
    )

    test_probabilities = predict_probabilities(model, x_test)
    test_result = evaluate_predictions(
        y_true=y_test,
        probabilities=test_probabilities,
        labels=encoder.classes_.astype(str).tolist(),
        output_dir=evaluation_dir,
        split_name="test",
        sample_ids=test_ids,
        threshold=threshold,
    )

    if len(encoder.classes_) == 2:
        # threshold-free metrics (AP, calibration slope) available to promotion policies; locked metrics unchanged
        validation_result.metrics.update(probability_metrics(y_val, positive_class_probabilities(val_probabilities)))
        test_result.metrics.update(probability_metrics(y_test, positive_class_probabilities(test_probabilities)))

    model_path = save_model(model, output_path / "model.pkl")
    (output_path / "selected_genes.txt").write_text("\n".join(selected_genes) + "\n", encoding="utf-8")

    shap_ranking, shap_path = export_shap_rankings(
        model=model,
        background=x_train,
        evaluation=x_val,
        output_dir=output_path / "shap",
        background_labels=pd.Series(y_train, index=x_train.index),
        evaluation_labels=pd.Series(y_val, index=x_val.index),
    )

    cv_metrics = cv_se = None
    if policy is not None and policy.source == "cv":
        # cross-validation on the development samples (train + validation); the test partition is never touched
        dev_ids = train_ids + val_ids
        y_dev = encoder.transform(phenotype.loc[dev_ids, label_column].astype(str))
        cv_metrics, cv_se = cross_validated_metrics(expression.loc[dev_ids], y_dev, model_family, top_k_genes,
                                                    excluded_genes, folds=cv_folds, random_seed=random_seed)

    runtime_seconds = perf_counter() - start
    candidate = {"model_family": model_family, "feature_count": len(selected_genes), "run_type": run_type,
                 "validation_metrics": validation_result.metrics, "cv_metrics": cv_metrics, "cv_se": cv_se}
    status = determine_acceptance(log_path, dataset_id, validation_result.metrics, run_type=run_type,
                                  policy=policy, candidate_entry=candidate)
    if policy is not None:
        provenance["policy_path"] = str(policy_path)
        provenance["policy_sha256"] = sha256_file(policy_path)
    record = {
        "timestamp": utc_now(),
        "dataset_id": dataset_id,
        "run_name": run_label,
        "run_type": run_type,
        "model_family": model_family,
        "feature_count": len(selected_genes),
        "excluded_genes": feature_result.excluded_genes,
        "validation_metrics": validation_result.metrics,
        "test_metrics": test_result.metrics,
        "runtime_seconds": round(runtime_seconds, 4),
        "notes": notes,
        "status": status,
        "output_dir": str(output_path),
        "prepared_dir": str(prepared_dir),
        "selected_genes_path": str(output_path / "selected_genes.txt"),
        "model_path": str(model_path),
        "shap_top_genes_path": str(shap_path),
        "git_commit_hash": provenance["git_commit_hash"],
        "provenance": provenance,
    }
    if cv_metrics is not None:
        record.update({"cv_metrics": cv_metrics, "cv_se": cv_se, "cv_folds": cv_folds})
    append_result(record, log_path=log_path)

    summary = TrainingRunSummary(
        run_name=run_label,
        run_type=run_type,
        model_family=model_family,
        dataset_id=dataset_id,
        output_dir=str(output_path),
        feature_count=len(selected_genes),
        status=status,
        runtime_seconds=runtime_seconds,
        validation_metrics=validation_result.metrics,
        test_metrics=test_result.metrics,
    )
    (output_path / "run_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    shap_ranking.to_csv(output_path / "shap" / "top50_preview.csv", index=False)
    return summary


def run_baseline(
    *,
    prepared_dir: str | Path,
    output_dir: str | Path,
    top_k_genes: int = 1000,
    log_path: str | Path = "results_log.json",
    random_seed: int = 42,
    **locked_options: Any,
) -> TrainingRunSummary:
    return run_training(
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        model_family="logistic_regression",
        top_k_genes=top_k_genes,
        run_name="baseline_logistic_regression",
        baseline=True,
        log_path=log_path,
        random_seed=random_seed,
        notes="Baseline logistic regression required by the locked evaluation policy.",
        run_type="baseline",
        **locked_options,
    )
