from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data_adapters.geo_loader import load_geo_dataset
from src.data_adapters.local_loader import load_local_dataset
from src.data_adapters.recount3_loader import load_recount3_dataset
from src.utils.io import PreparedDataset, save_prepared_dataset
from src.utils.validation import (
    auto_normalize_expression,
    create_frozen_splits,
    validate_and_align_dataset,
    validate_output_dir,
)


def _log_progress(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def _normalize_filter_values(values: Any) -> set[str]:
    if isinstance(values, (list, tuple, set)):
        return {str(value).strip() for value in values}
    return {str(values).strip()}


def _apply_phenotype_filters(
    expression: pd.DataFrame,
    phenotype: pd.DataFrame,
    *,
    phenotype_filters: dict[str, Any] | None = None,
    exclude_title_suffixes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_expression = expression.copy()
    filtered_phenotype = phenotype.copy()

    if phenotype_filters:
        for column, accepted_values in phenotype_filters.items():
            if column not in filtered_phenotype.columns:
                raise ValueError(f"Phenotype filter column {column!r} was not found")
            normalized = _normalize_filter_values(accepted_values)
            mask = filtered_phenotype[column].astype(str).str.strip().isin(normalized)
            filtered_phenotype = filtered_phenotype.loc[mask]

    if exclude_title_suffixes and "title" in filtered_phenotype.columns:
        title_series = filtered_phenotype["title"].astype(str)
        suffixes = tuple(suffix.lower() for suffix in exclude_title_suffixes)
        mask = ~title_series.str.lower().str.endswith(suffixes)
        filtered_phenotype = filtered_phenotype.loc[mask]

    shared_ids = filtered_expression.index.intersection(filtered_phenotype.index)
    filtered_expression = filtered_expression.loc[shared_ids]
    filtered_phenotype = filtered_phenotype.loc[shared_ids]
    return filtered_expression, filtered_phenotype


def _apply_label_mapping(
    phenotype: pd.DataFrame,
    *,
    label_column: str,
    label_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not label_mapping:
        return phenotype
    if label_column not in phenotype.columns:
        raise ValueError(f"Label column {label_column!r} was not found for label mapping")

    normalized_mapping = {str(key).strip(): str(value).strip() for key, value in label_mapping.items()}
    mapped = phenotype.copy()
    source = mapped[label_column].astype(str).str.strip()
    translated = source.map(normalized_mapping)
    if translated.isna().any():
        missing = sorted(source[translated.isna()].unique().tolist())
        raise ValueError(f"Label mapping did not define values for: {missing}")
    mapped[label_column] = translated.astype(str)
    return mapped


def prepare_dataset(
    *,
    input_mode: str,
    output_dir: str | Path,
    label_column: str,
    normalization: str = "auto",
    random_seed: int = 42,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    overwrite: bool = False,
    phenotype_filters: dict[str, Any] | None = None,
    exclude_title_suffixes: list[str] | None = None,
    label_mapping: dict[str, str] | None = None,
    **kwargs: Any,
) -> PreparedDataset:
    validate_output_dir(output_dir, overwrite=overwrite)
    _log_progress(f"starting prepare for input_mode={input_mode} output_dir={output_dir}")

    if input_mode == "local":
        _log_progress("loading local dataset")
        bundle = load_local_dataset(
            expression_path=kwargs["expression_path"],
            phenotype_path=kwargs["phenotype_path"],
            label_column=label_column,
            sample_id_column=kwargs.get("sample_id_column", "sample_id"),
            dataset_id=kwargs.get("dataset_id"),
        )
    elif input_mode == "geo":
        _log_progress(f"loading GEO dataset accession={kwargs['accession']}")
        bundle = load_geo_dataset(
            accession=kwargs["accession"],
            label_column=label_column,
            supplementary_urls=kwargs.get("supplementary_urls"),
        )
    elif input_mode == "recount3":
        _log_progress(f"loading recount3 dataset project={kwargs['project']}")
        bundle = load_recount3_dataset(
            project=kwargs["project"],
            label_column=label_column,
            organism=kwargs.get("organism", "human"),
            project_home=kwargs.get("project_home", "data_sources/sra"),
            annotation=kwargs.get("annotation", "G026"),
        )
    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")

    dataset_id_override = kwargs.get("dataset_id")
    if dataset_id_override:
        bundle.metadata = {**bundle.metadata, "dataset_id": dataset_id_override}

    _log_progress("applying phenotype filters and validating alignment")
    expression, phenotype = _apply_phenotype_filters(
        bundle.expression,
        bundle.phenotype,
        phenotype_filters=phenotype_filters,
        exclude_title_suffixes=exclude_title_suffixes,
    )
    phenotype = _apply_label_mapping(phenotype, label_column=label_column, label_mapping=label_mapping)
    expression, phenotype = validate_and_align_dataset(expression, phenotype, label_column=label_column)
    _log_progress(f"validated dataset with {expression.shape[0]} samples and {expression.shape[1]} genes")
    expression = auto_normalize_expression(expression, normalization)
    _log_progress(f"applied normalization={normalization}")
    phenotype = phenotype.copy()
    phenotype[label_column] = phenotype[label_column].astype(str)

    prepared = PreparedDataset(
        bundle=bundle.__class__(
            expression=expression,
            phenotype=phenotype,
            metadata={
                **bundle.metadata,
                "normalization": normalization,
                "phenotype_filters": phenotype_filters or {},
                "exclude_title_suffixes": exclude_title_suffixes or [],
                "label_mapping": label_mapping or {},
            },
            label_column=label_column,
        ),
        splits=create_frozen_splits(phenotype[label_column], val_fraction=val_fraction, test_fraction=test_fraction, seed=random_seed),
        split_seed=random_seed,
        normalization=normalization,
    )
    save_prepared_dataset(prepared, output_dir)
    _log_progress("saved prepared dataset and frozen splits")
    return prepared


def prepared_summary(prepared: PreparedDataset) -> dict[str, Any]:
    return {
        "dataset_id": prepared.bundle.dataset_id,
        "n_samples": int(prepared.bundle.expression.shape[0]),
        "n_genes": int(prepared.bundle.expression.shape[1]),
        "label_column": prepared.bundle.label_column,
        "split_sizes": {key: len(value) for key, value in prepared.splits.items()},
        "normalization": prepared.normalization,
    }
