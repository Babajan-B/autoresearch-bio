from pathlib import Path

import pandas as pd
import pytest

from src.prepare import prepare_dataset
from src.utils.io import load_prepared_dataset
from src.utils.validation import orient_expression_table


def test_prepare_creates_locked_split_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepared"
    prepared = prepare_dataset(
        input_mode="local",
        expression_path="tests/fixtures/local_expression.tsv",
        phenotype_path="tests/fixtures/local_phenotype.tsv",
        label_column="label",
        output_dir=output_dir,
        normalization="none",
    )

    loaded = load_prepared_dataset(output_dir)
    assert prepared.bundle.expression.shape == loaded.bundle.expression.shape
    assert set(loaded.splits) == {"train", "val", "test"}
    assert sum(len(values) for values in loaded.splits.values()) == 12

    with pytest.raises(FileExistsError):
        prepare_dataset(
            input_mode="local",
            expression_path="tests/fixtures/local_expression.tsv",
            phenotype_path="tests/fixtures/local_phenotype.tsv",
            label_column="label",
            output_dir=output_dir,
        )


def test_orient_expression_table_preserves_gene_names_from_unnamed_first_column() -> None:
    table = pd.DataFrame(
        {
            "Unnamed: 0": ["ESR1", "PGR", "ERBB2"],
            "sample_a": [1.0, 2.0, 3.0],
            "sample_b": [4.0, 5.0, 6.0],
        }
    )

    oriented = orient_expression_table(table, ["sample_a", "sample_b"])

    assert oriented.index.tolist() == ["sample_a", "sample_b"]
    assert oriented.columns.tolist() == ["ESR1", "PGR", "ERBB2"]


def test_prepare_applies_label_mapping(tmp_path: Path) -> None:
    output_dir = tmp_path / "mapped"
    prepared = prepare_dataset(
        input_mode="local",
        expression_path="tests/fixtures/local_expression.tsv",
        phenotype_path="tests/fixtures/local_phenotype.tsv",
        label_column="condition",
        label_mapping={"tumor": "1", "normal": "0"},
        output_dir=output_dir,
        normalization="none",
    )

    labels = prepared.bundle.phenotype["condition"].astype(str).unique().tolist()
    assert sorted(labels) == ["0", "1"]
