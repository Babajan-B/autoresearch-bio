from pathlib import Path

from src.external_validation import run_external_validation
from src.prepare import prepare_dataset
from src.train import run_baseline


def test_external_validation_smoke(tmp_path: Path) -> None:
    reference_prepared_dir = tmp_path / "reference_prepared"
    external_prepared_dir = tmp_path / "external_prepared"
    reference_run_dir = tmp_path / "reference_baseline"
    output_dir = tmp_path / "external_eval"
    log_path = tmp_path / "results_log.json"
    log_path.write_text("[]\n", encoding="utf-8")

    prepare_dataset(
        input_mode="local",
        expression_path="tests/fixtures/local_expression.tsv",
        phenotype_path="tests/fixtures/local_phenotype.tsv",
        label_column="label",
        output_dir=reference_prepared_dir,
        normalization="none",
    )
    prepare_dataset(
        input_mode="local",
        expression_path="tests/fixtures/local_expression.tsv",
        phenotype_path="tests/fixtures/local_phenotype.tsv",
        label_column="label",
        output_dir=external_prepared_dir,
        normalization="none",
    )
    run_baseline(
        prepared_dir=reference_prepared_dir,
        output_dir=reference_run_dir,
        log_path=log_path,
    )

    summary = run_external_validation(
        reference_run_dir=reference_run_dir,
        reference_prepared_dir=reference_prepared_dir,
        external_prepared_dir=external_prepared_dir,
        output_dir=output_dir,
        run_name="local_external_smoke",
    )

    assert summary.external_dataset_id == "local_expression"
    assert summary.shared_gene_count == summary.feature_count
    assert (output_dir / "evaluation" / "external_predictions.csv").exists()
