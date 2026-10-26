"""Selection study: how much does the way the reported model is chosen change what is reported?

For one endpoint: a stratified "vault" is sealed once and never seen by any selection strategy. The rest is split
R times into development and test sets. Every declared candidate (model family x gene count) is scored by
stratified cross-validation on the development set; the fold-1 model doubles as the single-validation-split model
of the legacy loop. Each candidate is refit on the whole development set and scored on test, vault and, when the
endpoint has one, an independent external cohort. Strategies are then compared on the same fitted models.

Expression inputs are dicts of feature spaces (for example "genes" and "hallmark" pathway scores) with the same
samples; a candidate's "feature_space" (default "genes") says which one it uses.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.features import select_top_variable_genes
from src.metrics import probability_metrics
from src.models import build_model, positive_class_probabilities
from src.promotion import Policy, choose

POST_HOC = "post_hoc_test"


def _fit_score(x: pd.DataFrame, y: np.ndarray, family: str, k: int, seed: int,
               evaluations: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> dict[str, dict[str, float]]:
    genes = select_top_variable_genes(x, k).selected_genes  # training samples only
    model = build_model(family, random_seed=seed).fit(x[genes], y)
    return {name: probability_metrics(yy, positive_class_probabilities(model.predict_proba(xx[genes])))
            for name, (xx, yy) in evaluations.items()}


def candidate_id(config: dict[str, Any]) -> str:
    return f"{config['model_family']}|{config.get('feature_space', 'genes')}|{int(config['top_k_genes'])}"


def run_repeat(spaces: dict[str, pd.DataFrame], dev: pd.Index, y_dev: np.ndarray, test: pd.Index, y_test: np.ndarray,
               holdouts: dict[str, tuple[dict[str, pd.DataFrame], np.ndarray]], grid: list[dict[str, Any]],
               folds: int, seed: int) -> pd.DataFrame:
    """One development/test partition: CV, fold-1 and refit scores for every candidate (long format)."""
    rows = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_indices = list(splitter.split(np.zeros(len(y_dev)), y_dev))
    for config in grid:
        family, k = config["model_family"], int(config["top_k_genes"])
        space = config.get("feature_space", "genes")
        x_dev, x_test = spaces[space].loc[dev], spaces[space].loc[test]
        held = {name: (x[space], y) for name, (x, y) in holdouts.items()}
        config = dict(config, candidate=candidate_id(config), feature_space=space)
        per_fold = []
        for fold, (tr, va) in enumerate(fold_indices):
            evaluations = {"validation": (x_dev.iloc[va], y_dev[va])}
            if fold == 0:  # the legacy loop's train/validation model is also scored on test and hold-outs
                evaluations |= {"test": (x_test, y_test)} | held
            scores = _fit_score(x_dev.iloc[tr], y_dev[tr], family, k, seed, evaluations)
            per_fold.append(scores["validation"])
            if fold == 0:
                rows += [dict(config, model="fold1", set=s, **m) for s, m in scores.items()]
        cv = pd.DataFrame(per_fold)
        rows.append(dict(config, model="cv", set="cv_mean", **cv.mean().to_dict()))
        rows.append(dict(config, model="cv", set="cv_se", **(cv.std(ddof=1) / np.sqrt(len(cv))).to_dict()))
        scores = _fit_score(x_dev, y_dev, family, k, seed, {"test": (x_test, y_test)} | held)
        rows += [dict(config, model="refit", set=s, **m) for s, m in scores.items()]
    return pd.DataFrame(rows)


def apply_strategies(scores: pd.DataFrame, grid: list[dict[str, Any]], policies: list[Policy],
                     baseline: dict[str, Any]) -> pd.DataFrame:
    """Which candidate each strategy picks in one repeat, and what it would report versus what holds up."""
    def table(model: str, set_name: str) -> pd.DataFrame:
        return scores[(scores.model == model) & (scores.set == set_name)].set_index("candidate")

    configs = [candidate_id(c) for c in grid]
    base = candidate_id(baseline)

    def candidates(model: str, set_name: str, se: pd.DataFrame | None = None) -> list[dict[str, Any]]:
        t = table(model, set_name)
        return [{"model_family": c.split("|")[0], "feature_count": int(c.split("|")[2]), "is_baseline": c == base,
                 "metrics": t.loc[c].to_dict(), "se": se.loc[c].to_dict() if se is not None else {}} for c in configs]

    picks: dict[str, tuple[int, str]] = {}
    picks[POST_HOC] = (int(np.argmax([table("refit", "test").loc[c, "auroc"] for c in configs])), "refit")
    for policy in policies:
        if policy.source == "validation":
            picks[policy.name] = (choose(candidates("fold1", "validation"), policy), "fold1")
        else:
            picks[policy.name] = (choose(candidates("cv", "cv_mean", table("cv", "cv_se")), policy), "refit")
    picks["baseline_only"] = (configs.index(base), "refit")

    holdout_sets = [s for s in scores.set.unique() if s not in {"validation", "test", "cv_mean", "cv_se"}]
    rows = []
    for strategy, (index, model) in picks.items():
        chosen = configs[index]
        row = {"strategy": strategy, "candidate": chosen, "model_family": chosen.split("|")[0],
               "feature_space": chosen.split("|")[1], "top_k_genes": int(chosen.split("|")[2]),
               "reported_test_auroc": table(model, "test").loc[chosen, "auroc"]}
        for s in holdout_sets:
            held = table(model, s)
            for metric in ("auroc", "average_precision", "brier_score", "calibration_slope", "calibration_in_the_large"):
                row[f"{s}_{metric}"] = held.loc[chosen, metric]
            # regret: shortfall from the best candidate on this hold-out among the same kind of fitted models
            row[f"{s}_auroc_regret"] = held["auroc"].max() - held.loc[chosen, "auroc"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_endpoint(spaces: dict[str, pd.DataFrame] | pd.DataFrame, labels: pd.Series, *, grid, policies, baseline,
                 repeats: int, folds: int, vault_fraction: float, test_fraction: float, vault_seed: int,
                 repeat_seed: int, external: tuple[dict[str, pd.DataFrame] | pd.DataFrame, pd.Series] | None = None,
                 progress=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    spaces = spaces if isinstance(spaces, dict) else {"genes": spaces}
    index = spaces["genes"].index
    labels = labels.dropna().astype(int)
    labels = labels[labels.index.isin(index)]
    rest, vault = train_test_split(labels.index, test_size=vault_fraction, stratify=labels, random_state=vault_seed)
    holdouts = {"vault": ({k: v.loc[vault] for k, v in spaces.items()}, labels.loc[vault].to_numpy())}
    if external is not None:
        ext_spaces, ext_y = external
        ext_spaces = ext_spaces if isinstance(ext_spaces, dict) else {"genes": ext_spaces}
        ext_y = ext_y.dropna().astype(int)
        ext_y = ext_y[ext_y.index.isin(ext_spaces["genes"].index)]
        holdouts["external"] = ({k: v.loc[ext_y.index] for k, v in ext_spaces.items()}, ext_y.to_numpy())
    all_scores, all_picks = [], []
    for r in range(repeats):
        seed = repeat_seed + r
        dev, test = train_test_split(rest, test_size=test_fraction, stratify=labels.loc[rest], random_state=seed)
        scores = run_repeat(spaces, dev, labels.loc[dev].to_numpy(), test, labels.loc[test].to_numpy(),
                            holdouts, grid, folds, seed)
        picks = apply_strategies(scores, grid, policies, baseline)
        all_scores.append(scores.assign(repeat=r))
        all_picks.append(picks.assign(repeat=r))
        progress(f"repeat {r + 1}/{repeats} done")
    return pd.concat(all_scores, ignore_index=True), pd.concat(all_picks, ignore_index=True)


def policies_from_config(entries: list[dict[str, Any]]) -> list[Policy]:
    return [Policy(**entry) for entry in entries]


def policy_dicts(policies: list[Policy]) -> list[dict[str, Any]]:
    return [asdict(p) for p in policies]


def write_outputs(out_dir: str | Path, scores: pd.DataFrame, picks: pd.DataFrame) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"scores": out / "candidate_scores.csv.gz", "picks": out / "strategy_picks.csv"}
    scores.to_csv(paths["scores"], index=False)
    picks.to_csv(paths["picks"], index=False)
    return paths
