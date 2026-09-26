from pathlib import Path

from src.data_adapters.local_loader import load_local_dataset


def test_local_loader_aligns_expression_and_phenotype() -> None:
    fixtures = Path("tests/fixtures")
    bundle = load_local_dataset(
        expression_path=fixtures / "local_expression.tsv",
        phenotype_path=fixtures / "local_phenotype.tsv",
        label_column="label",
    )

    assert bundle.expression.shape == (12, 8)
    assert bundle.phenotype.shape[0] == 12
    assert bundle.label_column == "label"
    assert bundle.expression.index.tolist()[0] == "S01"
