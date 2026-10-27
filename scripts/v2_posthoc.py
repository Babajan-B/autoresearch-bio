"""Post hoc analyses (not in the declaration; reported as such). Writes results/v2/posthoc/.

P1  Nadeau-Bengio corrected resampled t-test of selection bias per strategy and endpoint.
P2  Tolerance sweep of the one-SE rule (0.25-2 x SE) from the stored cross-validation results (no refitting).
P3  Chosen models refitted for every repeat (TCGA endpoints; baseline, CV AUROC, CV AUROC 1-SE simplest, CV Brier
    1-SE simplest): ECE with 10 equal-frequency bins; prior-odds correction from the balanced training prior (all
    models are class-weighted) to the SCAN-B prevalence and to the TCGA prevalence; net benefit at thresholds
    0.10-0.50. Refitted Brier scores must equal the logged ones.
P4  Baseline logistic regression without class weighting, and class-weighted with Platt scaling (5-fold, development).
P5  Supervised, label-aware gene filter (ANOVA F on development samples) against the declared variance filter for
    HER2, grade 3 and Ki67 (logistic regression and random forest; 500 and 2,000 genes).

Usage: python -m scripts.v2_posthoc --data-root ../.. [--only P1,P2,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import select_top_variable_genes
from src.metrics import calibration_slope
from src.models import build_model, positive_class_probabilities
from src.modules import load_gmt, rank_scores
from src.promotion import Policy, choose

OUT = Path("results/v2/posthoc")
TCGA_ENDPOINTS = ["er", "pgr", "her2", "node"]
P3_STRATEGIES = ["baseline_only", "cv_auroc", "cv_auroc_one_se_simplest", "cv_brier_one_se_simplest"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def ece(y, p, bins=10):
    """Expected calibration error with equal-frequency bins: weighted mean |observed - predicted|."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    f = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q")
    return float((f.size() / len(y) * (f.y.mean() - f.p.mean()).abs()).sum())


def net_benefit(y, p, t):
    pos = p >= t
    n = len(y)
    tp, fp = np.sum(pos & (y == 1)), np.sum(pos & (y == 0))
    return tp / n - fp / n * t / (1 - t)


def prob_metrics(y, p):
    return {"auroc": roc_auc_score(y, p), "brier": float(np.mean((p - y) ** 2)), "slope": calibration_slope(y, p),
            "citl": float(np.mean(p) - np.mean(y)), "ece": ece(y, p)}


# ------------------------------------------------------------------ P1
def p1_corrected_t(study):
    rows = []
    test_over_train = study["design"]["test_fraction"] / (1 - study["design"]["test_fraction"])
    for e in study["endpoints"]:
        picks = pd.read_csv(f"results/v2/{e}/strategy_picks.csv")
        scores = pd.read_csv(f"results/v2/{e}/candidate_scores.csv.gz")
        mean_test = scores[scores.set == "test"].groupby(["model", "candidate"]).auroc.mean()
        for s, g in picks.groupby("strategy", sort=False):
            kind = "fold1" if s == "legacy_auroc_single_split" else "refit"
            bias = g.reported_test_auroc.to_numpy() - np.array([mean_test.loc[(kind, c)] for c in g.candidate])
            j = len(bias)
            var = np.var(bias, ddof=1)
            se = np.sqrt((1 / j + test_over_train) * var)
            t = bias.mean() / se if se > 0 else np.nan
            p = 2 * stats.t.sf(abs(t), df=j - 1) if se > 0 else np.nan
            rows.append({"endpoint": e, "strategy": s, "mean_bias": bias.mean(), "corrected_se": se, "t": t, "p": p,
                         "naive_p": stats.ttest_1samp(bias, 0).pvalue if var > 0 else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ P2
def p2_tolerance_sweep(study):
    from src.selection_study import candidate_id
    grid = [candidate_id(c) for c in study["grid"]]
    base = candidate_id(study["baseline"])
    rows = []
    for e in study["endpoints"]:
        sc = pd.read_csv(f"results/v2/{e}/candidate_scores.csv.gz")
        for r, g in sc.groupby("repeat"):
            mean = g[(g.model == "cv") & (g.set == "cv_mean")].set_index("candidate")
            se = g[(g.model == "cv") & (g.set == "cv_se")].set_index("candidate")
            vault = g[(g.model == "refit") & (g.set == "vault")].set_index("candidate").auroc
            for eps in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
                cands = [{"model_family": c.split("|")[0], "feature_count": int(c.split("|")[2]), "is_baseline": c == base,
                          "metrics": mean.loc[c].to_dict(), "se": {"auroc": eps * se.loc[c, "auroc"]}} for c in grid]
                pick = grid[choose(cands, Policy(name="s", source="cv", tolerance="one_se",
                                                 tie_breakers=["complexity", "feature_count"]))]
                rows.append({"endpoint": e, "repeat": r, "eps": eps, "candidate": pick,
                             "logistic": pick.startswith("logistic"), "genes": int(pick.split("|")[2]),
                             "vault_auroc": vault.loc[pick], "vault_regret": vault.max() - vault.loc[pick]})
    d = pd.DataFrame(rows)
    stab = d.groupby(["endpoint", "eps"]).candidate.agg(lambda c: c.value_counts().iloc[0] / len(c)).rename("stability")
    return d.groupby(["endpoint", "eps"]).agg(share_logistic=("logistic", "mean"), mean_features=("genes", "mean"),
                                               vault_auroc=("vault_auroc", "mean"),
                                               vault_regret=("vault_regret", "mean")).join(stab).reset_index()


# ------------------------------------------------------------------ data
def load(root, study):
    scanb = pd.read_pickle(root / study["data"]["scanb_expression"])
    tcga = pd.read_pickle(root / study["data"]["tcga_expression"])
    sets = load_gmt(study["gene_sets"]["hallmark"])
    spaces = {"genes": (scanb, tcga), "hallmark": (rank_scores(scanb, sets), rank_scores(tcga, sets))}
    return spaces, pd.read_csv(root / study["data"]["scanb_labels"], index_col=0), \
        pd.read_csv(root / study["data"]["tcga_labels"], index_col=0)


def partitions(labels, e, d, r):
    y = labels[e].dropna().astype(int)
    rest, vault = train_test_split(y.index, test_size=d["vault_fraction"], stratify=y, random_state=d["vault_seed"])
    seed = d["repeat_seed"] + r
    dev, test = train_test_split(rest, test_size=d["test_fraction"], stratify=y.loc[rest], random_state=seed)
    return y, dev, vault, seed


# ------------------------------------------------------------------ P3 + P4
def p3_p4(root, study):
    spaces, labels, tlabels = load(root, study)
    d = study["design"]
    m_rows, nb_rows, checks = [], [], []
    for e in TCGA_ENDPOINTS:
        picks = pd.read_csv(f"results/v2/{e}/strategy_picks.csv")
        ty = tlabels[e].dropna().astype(int)
        ty = ty[ty.index.isin(spaces["genes"][1].index)]
        yt = ty.to_numpy()
        for r in range(d["repeats"]):
            y, dev, _, seed = partitions(labels, e, d, r)
            ydev = y.loc[dev].to_numpy()
            pi_scanb, pi_tcga = ydev.mean(), yt.mean()
            pr = picks[picks.repeat == r].set_index("strategy")
            cache = {}
            for s in P3_STRATEGIES:
                cand = pr.loc[s, "candidate"]
                if cand not in cache:
                    family, space, k = cand.split("|")
                    xd, xt = spaces[space][0].loc[dev], spaces[space][1].loc[ty.index]
                    genes = select_top_variable_genes(xd, int(k)).selected_genes
                    model = build_model(family, random_seed=seed).fit(xd[genes], ydev)
                    cache[cand] = positive_class_probabilities(model.predict_proba(xt[genes]))
                p = cache[cand]
                checks.append(abs(np.mean((p - yt) ** 2) - pr.loc[s, "external_brier_score"]))
                # class-weighted models carry a balanced (0.5) prior; shift it to SCAN-B, then to TCGA prevalence
                versions = {"raw": p, "prior_scanb": expit(logit(p) + logit(pi_scanb)),
                            "prior_tcga": expit(logit(p) + logit(pi_tcga))}
                for v, pv in versions.items():
                    m_rows.append({"endpoint": e, "repeat": r, "strategy": s, "candidate": cand, "version": v,
                                   **prob_metrics(yt, pv)})
                for t in np.round(np.arange(0.10, 0.51, 0.05), 2):
                    nb_rows.append({"endpoint": e, "repeat": r, "strategy": s, "threshold": t,
                                    "net_benefit": net_benefit(yt, p, t),
                                    "net_benefit_prior_tcga": net_benefit(yt, versions["prior_tcga"], t),
                                    "treat_all": pi_tcga - (1 - pi_tcga) * t / (1 - t)})
            # P4: baseline logistic regression without class weighting, and with Platt scaling
            xd, xt = spaces["genes"][0].loc[dev], spaces["genes"][1].loc[ty.index]
            genes = select_top_variable_genes(xd, 2000).selected_genes
            unweighted = Pipeline([("scaler", StandardScaler()),
                                   ("model", LogisticRegression(max_iter=5000, random_state=seed))]).fit(xd[genes], ydev)
            platt = CalibratedClassifierCV(build_model("logistic_regression", random_seed=seed), method="sigmoid",
                                           cv=5).fit(xd[genes], ydev)
            for v, mdl in (("lr_unweighted", unweighted), ("lr_weighted_platt", platt)):
                m_rows.append({"endpoint": e, "repeat": r, "strategy": "baseline_only", "candidate": "logistic_regression|genes|2000",
                               "version": v, **prob_metrics(yt, mdl.predict_proba(xt[genes])[:, 1])})
            print(f"[P3/P4] {e} repeat {r + 1} done", flush=True)
    assert max(checks) < 1e-9, f"refitted models differ from the logged ones (max |dBrier| {max(checks)})"
    return pd.DataFrame(m_rows), pd.DataFrame(nb_rows)


# ------------------------------------------------------------------ P5
def p5_supervised_filter(root, study):
    spaces, labels, tlabels = load(root, study)
    d = study["design"]
    scanb, tcga = spaces["genes"]
    rows = []
    for e in ["her2", "nhg3", "ki67"]:
        for r in range(d["repeats"]):
            y, dev, vault, seed = partitions(labels, e, d, r)
            xd, ydev = scanb.loc[dev], y.loc[dev].to_numpy()
            f = pd.Series(f_classif(xd, ydev)[0], index=xd.columns).fillna(0)
            variance_rank = xd.var().rank(ascending=False)
            for k in (500, 2000):
                for filt, genes in (("variance", select_top_variable_genes(xd, k).selected_genes),
                                    ("supervised_F", f.nlargest(k).index.tolist())):
                    gset = set(genes)
                    for family in ("logistic_regression", "random_forest"):
                        model = build_model(family, random_seed=seed).fit(xd[genes], ydev)
                        row = {"endpoint": e, "repeat": r, "filter": filt, "k": k, "model": family,
                               "has_ERBB2": "ERBB2" in gset, "has_MKI67": "MKI67" in gset,
                               "vault_auroc": roc_auc_score(y.loc[vault], positive_class_probabilities(
                                   model.predict_proba(scanb.loc[vault, genes])))}
                        if e in tlabels:
                            ty = tlabels[e].dropna().astype(int)
                            ty = ty[ty.index.isin(tcga.index)]
                            row["tcga_auroc"] = roc_auc_score(ty, positive_class_probabilities(
                                model.predict_proba(tcga.loc[ty.index, genes])))
                        rows.append(row)
            print(f"[P5] {e} repeat {r + 1} done", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--only", default="P1,P2,P3,P5")
    args = parser.parse_args()
    study = yaml.safe_load(Path("config/v2/study.yaml").read_text())
    root = Path(args.data_root)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = args.only.split(",")
    if "P1" in todo:
        p1_corrected_t(study).to_csv(OUT / "P1_corrected_t.csv", index=False)
    if "P2" in todo:
        p2_tolerance_sweep(study).to_csv(OUT / "P2_tolerance_sweep.csv", index=False)
    if "P3" in todo:
        m, nb = p3_p4(root, study)
        m.to_csv(OUT / "P3_P4_tcga_calibration.csv.gz", index=False)
        nb.to_csv(OUT / "P3_net_benefit.csv.gz", index=False)
    if "P5" in todo:
        p5_supervised_filter(root, study).to_csv(OUT / "P5_supervised_filter.csv", index=False)
    (OUT / "README.json").write_text(json.dumps({"note": "post hoc analyses, not declared", "parts": todo}, indent=1))


if __name__ == "__main__":
    main()
