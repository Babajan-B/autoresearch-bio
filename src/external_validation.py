from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.evaluate import evaluate_predictions
from src.models import load_model, model_family_name, predict_probabilities
from src.utils.io import ensure_dir, load_prepared_dataset, read_json, utc_now, write_json


@dataclass(slots=True)
class ExternalValidationSummary:
    run_name: str
    reference_run_dir: str
    reference_prepared_dir: str
    external_dataset_id: str
    output_dir: str
    model_family: str
    feature_count: int
    shared_gene_count: int
    missing_gene_count: int
    runtime_seconds: float
    external_metrics: dict[str, Any]


def _load_selected_genes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_threshold(reference_run_dir: Path) -> float | None:
    metrics = read_json(reference_run_dir / "evaluation" / "validation_metrics.json", default={}) or {}
    value = metrics.get("threshold")
    return None if value is None else float(value)


def _build_external_matrix(expression: pd.DataFrame, selected_genes: list[str]) -> tuple[pd.DataFrame, int, int]:
    external_columns = {str(column) for column in expression.columns}
    shared = [gene for gene in selected_genes if gene in external_columns]
    aligned = expression.reindex(columns=selected_genes, fill_value=0.0)
    return aligned, len(shared), len(selected_genes) - len(shared)


def run_external_validation(
    *,
    reference_run_dir: str | Path,
    reference_prepared_dir: str | Path,
    external_prepared_dir: str | Path,
    output_dir: str | Path,
    run_name: str,
    notes: str = "",
) -> ExternalValidationSummary:
    reference_run_path = Path(reference_run_dir)
    reference_prepared = load_prepared_dataset(reference_prepared_dir)
    external_prepared = load_prepared_dataset(external_prepared_dir)

    selected_genes = _load_selected_genes(reference_run_path / "selected_genes.txt")
    model = load_model(reference_run_path / "model.pkl")
    threshold = _load_threshold(reference_run_path)

    label_column = reference_prepared.bundle.label_column
    encoder = LabelEncoder()
    encoder.fit(reference_prepared.bundle.phenotype[label_column].astype(str))

    external_label_column = external_prepared.bundle.label_column
    external_labels = external_prepared.bundle.phenotype[external_label_column].astype(str)
    unseen = sorted(set(external_labels.unique()) - set(encoder.classes_))
    if unseen:
        raise ValueError(f"External dataset contains unseen labels: {unseen}")

    x_external, shared_gene_count, missing_gene_count = _build_external_matrix(
        external_prepared.bundle.expression,
        selected_genes,
    )
    y_external = encoder.transform(external_labels)

    output_path = ensure_dir(output_dir)
    start = perf_counter()
    probabilities = predict_probabilities(model, x_external)
    evaluation = evaluate_predictions(
        y_true=y_external,
        probabilities=probabilities,
        labels=encoder.classes_.astype(str).tolist(),
        output_dir=output_path / "evaluation",
        split_name="external",
        sample_ids=x_external.index.astype(str).tolist(),
        threshold=threshold,
    )
    runtime_seconds = perf_counter() - start

    summary = ExternalValidationSummary(
        run_name=run_name,
        reference_run_dir=str(reference_run_path),
        reference_prepared_dir=str(reference_prepared_dir),
        external_dataset_id=external_prepared.bundle.dataset_id,
        output_dir=str(output_path),
        model_family=model_family_name(model),
        feature_count=len(selected_genes),
        shared_gene_count=shared_gene_count,
        missing_gene_count=missing_gene_count,
        runtime_seconds=runtime_seconds,
        external_metrics=evaluation.metrics,
    )

    payload = asdict(summary)
    payload["created_at"] = utc_now()
    payload["notes"] = notes
    write_json(output_path / "external_validation_summary.json", payload)
    (output_path / "reference_run.txt").write_text(f"{reference_run_path}\n", encoding="utf-8")
    (output_path / "selected_genes.txt").write_text("\n".join(selected_genes) + "\n", encoding="utf-8")
    return summary
