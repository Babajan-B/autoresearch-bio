import json
from pathlib import Path

import pytest

from src.logging_utils import append_result, effective_status, read_results_log, verify_log
from src.prepare import prepare_dataset
from src.promotion import LEGACY_AUROC, Policy, choose
from src.train import run_baseline, run_training


def _entry(name, auroc, status="accepted", run_type="train"):
    return {"dataset_id": "d", "run_name": name, "run_type": run_type, "status": status,
            "validation_metrics": {"auroc": auroc}}


def test_hash_chain_detects_edits_and_superseded_is_derived(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    append_result(_entry("base", 0.8, run_type="baseline"), log)
    append_result(_entry("a", 0.85), log)
    append_result(_entry("b", 0.90), log)
    assert verify_log(log) == []
    assert [effective_status(read_results_log(log))[i] for i in range(3)] == ["accepted", "superseded", "accepted"]

    entries = json.loads(log.read_text())
    entries[1]["validation_metrics"]["auroc"] = 0.99  # silent edit of an earlier result
    log.write_text(json.dumps(entries))
    assert any("does not match entry_hash" in p for p in verify_log(log))

    entries[1]["entry_hash"] = "f" * 64  # re-hashing one entry breaks the next link
    log.write_text(json.dumps(entries))
    assert any("prev_hash" in p for p in verify_log(log))


def _cand(family, k, auroc, se=0.01, slope=1.0, baseline=False):
    return {"model_family": family, "feature_count": k, "is_baseline": baseline,
            "metrics": {"auroc": auroc, "macro_f1": 0.5, "brier_score": 0.1, "mcc": 0.5, "calibration_slope": slope},
            "se": {"auroc": se}}


def test_policies() -> None:
    pool = [_cand("logistic_regression", 1000, 0.940, baseline=True),
            _cand("extra_trees", 2000, 0.947, slope=1.8),
            _cand("random_forest", 2000, 0.944)]
    assert choose(pool, LEGACY_AUROC) == 1
    one_se = Policy(name="s", source="cv", tolerance="one_se", tie_breakers=["complexity", "feature_count"])
    assert choose(pool, one_se) == 0  # baseline is within one SE and simplest
    calibrated = Policy(name="c", constraints=[{"metric": "calibration_slope", "min": 0.7, "max": 1.4}])
    assert choose(pool, calibrated) == 2
    impossible = Policy(name="x", constraints=[{"metric": "auroc", "min": 0.99}])
    assert choose(pool, impossible) == 0  # nothing qualifies -> keep the baseline
    with pytest.raises(ValueError):
        Policy(name="bad", source="test")


def test_loop_with_cv_policy_and_provenance(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepare_dataset(input_mode="local", expression_path="tests/fixtures/local_expression.tsv",
                    phenotype_path="tests/fixtures/local_phenotype.tsv", label_column="label",
                    output_dir=prepared, normalization="none")
    policy = tmp_path / "policy.yaml"
    policy.write_text("name: one_se\nsource: cv\ntolerance: one_se\ntie_breakers: [complexity, feature_count]\n")
    declaration = tmp_path / "DECLARATION.md"
    declaration.write_text("plan")
    log = tmp_path / "log.json"
    opts = dict(log_path=log, policy_path=policy, declaration_path=declaration, cv_folds=2)
    run_baseline(prepared_dir=prepared, output_dir=tmp_path / "b", top_k_genes=6, **opts)
    run_training(prepared_dir=prepared, output_dir=tmp_path / "rf", model_family="random_forest",
                 run_name="rf", top_k_genes=6, **opts)
    entries = read_results_log(log)
    assert entries[1]["cv_metrics"]["auroc"] >= 0
    assert entries[1]["provenance"]["declaration_sha256"]
    assert verify_log(log) == []
    declaration.write_text("plan changed after the runs")
    assert any("changed since the run" in p for p in verify_log(log))


def test_selection_study_runs_and_post_hoc_is_most_optimistic() -> None:
    import numpy as np
    import pandas as pd

    from src.selection_study import apply_strategies, run_endpoint

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(160, 30)), index=[f"s{i}" for i in range(160)], columns=[f"g{j}" for j in range(30)])
    y = pd.Series((x["g0"] + rng.normal(scale=2, size=160) > 0).astype(int), index=x.index)
    grid = [{"model_family": "logistic_regression", "top_k_genes": 10}, {"model_family": "random_forest", "top_k_genes": 30}]
    policies = [Policy(name="legacy", tie_breakers=["brier_score"]), Policy(name="cv1se", source="cv", tolerance="one_se", tie_breakers=["complexity"])]
    scores, picks = run_endpoint(x, y, grid=grid, policies=policies, baseline=grid[0], repeats=2, folds=3,
                                 vault_fraction=0.25, test_fraction=0.25, vault_seed=1, repeat_seed=1,
                                 external=(x, y), progress=lambda m: None)
    assert set(picks.strategy) == {"post_hoc_test", "legacy", "cv1se", "baseline_only"}
    assert {"vault_auroc", "external_auroc", "vault_auroc_regret"} <= set(picks.columns)
    post_hoc = picks[picks.strategy == "post_hoc_test"].reported_test_auroc
    for strategy in ("cv1se", "baseline_only"):
        assert (post_hoc.values >= picks[picks.strategy == strategy].reported_test_auroc.values - 1e-12).all()


def test_pathway_scores_are_per_sample_and_study_accepts_spaces() -> None:
    import numpy as np
    import pandas as pd

    from src.modules import rank_scores
    from src.selection_study import run_endpoint

    rng = np.random.default_rng(1)
    genes = [f"g{j}" for j in range(40)]
    x = pd.DataFrame(rng.normal(size=(120, 40)), index=[f"s{i}" for i in range(120)], columns=genes)
    sets = {"up": genes[:10], "rest": genes[10:30], "too_small": genes[:3]}
    scores = rank_scores(x, sets)
    assert list(scores.columns) == ["up", "rest"]
    # scoring a single sample alone gives the same value: no cross-sample information
    assert np.allclose(rank_scores(x.iloc[[5]], sets).to_numpy(), scores.iloc[[5]].to_numpy())
    y = pd.Series((x[genes[:10]].mean(axis=1) > 0).astype(int), index=x.index)
    grid = [{"model_family": "logistic_regression", "top_k_genes": 10},
            {"model_family": "logistic_regression", "feature_space": "pathways", "top_k_genes": 2}]
    candidate_scores, picks = run_endpoint({"genes": x, "pathways": scores}, y, grid=grid, policies=[], baseline=grid[0], repeats=1,
                            folds=3, vault_fraction=0.25, test_fraction=0.25, vault_seed=1, repeat_seed=1,
                            progress=lambda m: None)
    assert set(candidate_scores.candidate) == {"logistic_regression|genes|10", "logistic_regression|pathways|2"}
    assert set(picks.feature_space) <= {"genes", "pathways"}


def test_declared_study_config_is_complete() -> None:
    import yaml

    from src.promotion import Policy
    from src.selection_study import candidate_id

    study = yaml.safe_load(Path("config/v2/study.yaml").read_text())
    assert study["endpoints"] == ["er", "pgr", "her2", "nhg3", "ki67", "luma_lumb", "node"]
    assert study["external_endpoints"] == ["er", "pgr", "her2", "node"]
    ids = [candidate_id(c) for c in study["grid"]]
    assert len(ids) == len(set(ids)) == 12 and candidate_id(study["baseline"]) in ids
    assert len([Policy(**p) for p in study["policies"]]) == 7
