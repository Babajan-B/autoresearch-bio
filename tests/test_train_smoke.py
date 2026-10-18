from pathlib import Path

from src.ablations import run_gene_ablation
from src.prepare import prepare_dataset
from src.train import run_baseline, run_training


def test_training_smoke_end_to_end(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    log_path = tmp_path / "results_log.json"
    log_path.write_text("[]", encoding="utf-8")

    prepare_dataset(
        input_mode="local",
        expression_path="tests/fixtures/local_expression.tsv",
        phenotype_path="tests/fixtures/local_phenotype.tsv",
        label_column="label",
        output_dir=prepared_dir,
        normalization="none",
    )

    baseline = run_baseline(
        prepared_dir=prepared_dir,
        output_dir=tmp_path / "baseline",
        log_path=log_path,
    )
    assert baseline.status == "accepted"

    training = run_training(
        prepared_dir=prepared_dir,
        output_dir=tmp_path / "rf_run",
        model_family="random_forest",
        run_name="rf_run",
        log_path=log_path,
        top_k_genes=6,
    )
    assert training.feature_count == 6

    ablation = run_gene_ablation(
        prepared_dir=prepared_dir,
        output_dir=tmp_path / "ablation",
        genes=["ESR1", "PGR", "ERBB2"],
        log_path=log_path,
    )
    assert ablation.run_type == "ablation"
