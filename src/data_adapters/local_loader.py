from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import DatasetBundle, read_table, safe_slug
from src.utils.validation import orient_expression_table


def load_local_dataset(
    expression_path: str | Path,
    phenotype_path: str | Path,
    label_column: str,
    sample_id_column: str = "sample_id",
    dataset_id: str | None = None,
) -> DatasetBundle:
    phenotype = read_table(phenotype_path)
    if sample_id_column not in phenotype.columns:
        raise ValueError(f"Expected phenotype table to contain sample ID column {sample_id_column!r}")
    phenotype = phenotype.set_index(sample_id_column)

    expression = read_table(expression_path)
    oriented_expression = orient_expression_table(expression, phenotype.index)

    metadata = {
        "dataset_id": dataset_id or safe_slug(Path(expression_path).stem),
        "source": "local",
        "expression_path": str(expression_path),
        "phenotype_path": str(phenotype_path),
    }
    return DatasetBundle(
        expression=oriented_expression,
        phenotype=phenotype,
        metadata=metadata,
        label_column=label_column,
    )
