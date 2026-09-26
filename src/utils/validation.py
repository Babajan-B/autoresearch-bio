from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


GEO_ACCESSION_RE = re.compile(r"^GSE\d+$", re.IGNORECASE)
RECOUNT3_PROJECT_RE = re.compile(r"^(S|E|D)RP\d+$", re.IGNORECASE)


def validate_geo_accession(accession: str) -> str:
    normalized = accession.strip().upper()
    if not GEO_ACCESSION_RE.fullmatch(normalized):
        raise ValueError(f"Expected a GEO series accession like GSE12345, received {accession!r}")
    return normalized


def validate_recount3_project(project: str) -> str:
    normalized = project.strip().upper()
    if not RECOUNT3_PROJECT_RE.fullmatch(normalized):
        raise ValueError(f"Expected a recount3 study ID like SRP009615, received {project!r}")
    return normalized


def _clean_index(values: pd.Index) -> pd.Index:
    return pd.Index([str(value).strip() for value in values], dtype="object")


def _ensure_first_column_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Expression table is empty")
    first_column = str(frame.columns[0]).strip().lower()
    id_like = {"sample", "sample_id", "gene", "gene_id", "symbol", "id", "id_ref", "gene_symbol"}
    if first_column in id_like or first_column.startswith("unnamed"):
        return frame.set_index(frame.columns[0])
    return frame


def orient_expression_table(expression: pd.DataFrame, sample_ids: list[str] | pd.Index | set[str]) -> pd.DataFrame:
    sample_id_set = {str(sample_id).strip() for sample_id in sample_ids}
    oriented = _ensure_first_column_index(expression.copy())
    oriented.index = _clean_index(pd.Index(oriented.index))
    oriented.columns = _clean_index(pd.Index(oriented.columns))

    row_overlap = len(sample_id_set.intersection(oriented.index))
    column_overlap = len(sample_id_set.intersection(oriented.columns))

    if column_overlap > row_overlap:
        oriented = oriented.transpose()
        oriented.index = _clean_index(pd.Index(oriented.index))
        oriented.columns = _clean_index(pd.Index(oriented.columns))
        row_overlap = len(sample_id_set.intersection(oriented.index))

    if row_overlap == 0:
        raise ValueError("Could not align expression table samples to phenotype sample IDs")

    oriented = oriented.loc[oriented.index.intersection(sample_id_set)]
    oriented = oriented.apply(pd.to_numeric, errors="coerce")
    if oriented.isna().any().any():
        raise ValueError("Expression table contains non-numeric values after orientation")
    if oriented.columns.duplicated().any():
        oriented = oriented.T.groupby(level=0).mean().T
    return oriented


def validate_and_align_dataset(
    expression: pd.DataFrame,
    phenotype: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_column not in phenotype.columns:
        raise ValueError(f"Label column {label_column!r} was not found in the phenotype table")

    aligned_expression = expression.copy()
    aligned_expression.index = _clean_index(pd.Index(aligned_expression.index))
    aligned_expression.columns = _clean_index(pd.Index(aligned_expression.columns))
    aligned_expression = aligned_expression.apply(pd.to_numeric, errors="coerce")
    if aligned_expression.isna().any().any():
        raise ValueError("Expression matrix contains missing or non-numeric values")

    aligned_phenotype = phenotype.copy()
    if aligned_phenotype.index.name is None and "sample_id" in aligned_phenotype.columns:
        aligned_phenotype = aligned_phenotype.set_index("sample_id")
    aligned_phenotype.index = _clean_index(pd.Index(aligned_phenotype.index))

    if aligned_expression.index.duplicated().any():
        raise ValueError("Expression matrix contains duplicated sample IDs")
    if aligned_phenotype.index.duplicated().any():
        raise ValueError("Phenotype table contains duplicated sample IDs")

    shared_ids = aligned_expression.index.intersection(aligned_phenotype.index)
    if len(shared_ids) < 4:
        raise ValueError("At least four aligned samples are required")

    aligned_expression = aligned_expression.loc[shared_ids].sort_index()
    aligned_phenotype = aligned_phenotype.loc[shared_ids].sort_index()

    if aligned_phenotype[label_column].isna().any():
        raise ValueError(f"Label column {label_column!r} contains missing values")
    if aligned_phenotype[label_column].nunique() < 2:
        raise ValueError("At least two label classes are required for classification")

    return aligned_expression, aligned_phenotype


def _fallback_split(labels: pd.Series, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for sample_id, label in labels.items():
        groups.setdefault(str(label), []).append(str(sample_id))

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for sample_ids in groups.values():
        sample_ids = list(sample_ids)
        rng.shuffle(sample_ids)
        for index, sample_id in enumerate(sample_ids):
            bucket = index % 3
            if bucket == 0:
                train_ids.append(sample_id)
            elif bucket == 1:
                val_ids.append(sample_id)
            else:
                test_ids.append(sample_id)

    if not val_ids and len(train_ids) > 1:
        val_ids.append(train_ids.pop())
    if not test_ids and len(train_ids) > 1:
        test_ids.append(train_ids.pop())

    return {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}


def create_frozen_splits(
    labels: pd.Series,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")

    sample_ids = labels.index.to_numpy()
    values = labels.to_numpy()
    class_counts = labels.value_counts()

    if len(labels) < 12 or class_counts.min() < 3:
        return _fallback_split(labels, seed)

    train_ids, temp_ids, _, temp_labels = train_test_split(
        sample_ids,
        values,
        test_size=val_fraction + test_fraction,
        stratify=values,
        random_state=seed,
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=seed,
    )

    return {
        "train": sorted(str(sample_id) for sample_id in train_ids),
        "val": sorted(str(sample_id) for sample_id in val_ids),
        "test": sorted(str(sample_id) for sample_id in test_ids),
    }


def auto_normalize_expression(expression: pd.DataFrame, method: str) -> pd.DataFrame:
    normalized = method.lower()
    if normalized == "none":
        return expression
    if normalized == "log1p":
        if (expression < 0).any().any():
            raise ValueError("log1p normalization requires non-negative expression values")
        return np.log1p(expression)
    if normalized != "auto":
        raise ValueError(f"Unsupported normalization method: {method}")

    if (expression < 0).any().any():
        return expression
    if float(expression.max().max()) > 50:
        return np.log1p(expression)
    return expression


def validate_output_dir(output_dir: str | Path, overwrite: bool = False) -> Path:
    path = Path(output_dir)
    manifest = path / "manifest.json"
    if manifest.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} already contains a prepared dataset; use overwrite to replace it")
    return path
