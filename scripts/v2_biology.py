"""Declared biological analyses B1-B3 of the v2 study (see config/v2/DECLARATION.md, Amendment 1).

Uses non-vault SCAN-B tumours (the vault is reconstructed with the declared seed and excluded) and TCGA-BRCA.
B4-B5 combine these outputs with results/v2 and are done in the figure script.
Usage: python -m scripts.v2_biology --data-root ../.. [--out results/v2/biology]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.modules import load_gmt, rank_scores

MARKERS = ["ESR1", "PGR", "FOXA1", "GATA3", "ERBB2", "GRB7", "STARD3", "PGAP3", "MIEN1", "MKI67", "TOP2A", "AURKA"]


def gene_aurocs(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """AUROC of every column for a binary label, via the rank-sum identity (fast for thousands of genes)."""
    ranks = x.rank(axis=0).to_numpy()
    positive = y.to_numpy() == 1
    n1, n0 = positive.sum(), (~positive).sum()
    return pd.Series((ranks[positive].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0), index=x.columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--out", default="results/v2/biology")
    args = parser.parse_args()
    study = yaml.safe_load(Path("config/v2/study.yaml").read_text())
    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scanb = pd.read_pickle(root / study["data"]["scanb_expression"])
    tcga = pd.read_pickle(root / study["data"]["tcga_expression"])
    scanb_labels = pd.read_csv(root / study["data"]["scanb_labels"], index_col=0)
    tcga_labels = pd.read_csv(root / study["data"]["tcga_labels"], index_col=0)
    gene_sets = load_gmt(study["gene_sets"]["hallmark"])
    scanb_paths, tcga_paths = rank_scores(scanb, gene_sets), rank_scores(tcga, gene_sets)
    design = study["design"]

    summary, marker_rows, top_rows, pathway_rows = {}, [], [], []
    for endpoint in study["endpoints"]:
        labels = scanb_labels[endpoint].dropna().astype(int)
        rest, _vault = train_test_split(labels.index, test_size=design["vault_fraction"], stratify=labels,
                                        random_state=design["vault_seed"])
        x, y = scanb.loc[rest], labels.loc[rest]
        auc = gene_aurocs(x, y)
        strength = (auc - 0.5).abs()
        variance_rank = x.var().rank(ascending=False)
        top100 = strength.nlargest(100).index
        summary[endpoint] = {
            "n": int(len(y)), "prevalence": float(y.mean()),
            "genes_auc_ge_0.70_or_le_0.30": int((strength >= 0.20).sum()),
            "max_gene_auc_distance": float(strength.max()),
            "top100_retained_by_variance_filter": {str(k): float((variance_rank.loc[top100] <= k).mean()) for k in (500, 2000, 5000)},
        }
        marker_rows += [{"endpoint": endpoint, "gene": g, "auroc": auc[g], "variance_rank": int(variance_rank[g])}
                        for g in MARKERS if g in auc]
        top_rows += [{"endpoint": endpoint, "rank": i + 1, "gene": g, "auroc": auc[g], "variance_rank": int(variance_rank[g])}
                     for i, g in enumerate(strength.nlargest(20).index)]
        path_scanb = gene_aurocs(scanb_paths.loc[rest], y)
        row = {"endpoint": endpoint, "cohort": "SCAN-B", **path_scanb.to_dict()}
        pathway_rows.append(row)
        if endpoint in study["external_endpoints"]:
            t = tcga_labels[endpoint].dropna().astype(int)
            t = t[t.index.isin(tcga.index)]
            path_tcga = gene_aurocs(tcga_paths.loc[t.index], t)
            pathway_rows.append({"endpoint": endpoint, "cohort": "TCGA", **path_tcga.to_dict()})
            rho = spearmanr(path_scanb, path_tcga.loc[path_scanb.index])
            summary[endpoint].update(tcga_n=int(len(t)), tcga_prevalence=float(t.mean()),
                                     pathway_auroc_spearman_scanb_vs_tcga=float(rho.statistic))
        # sanity check of the rank-sum identity on one gene
        g = strength.idxmax()
        assert np.isclose(auc[g], roc_auc_score(y, x[g]))

    pd.DataFrame(marker_rows).to_csv(out / "B2_marker_genes.csv", index=False)
    pd.DataFrame(top_rows).to_csv(out / "B1_top_genes.csv", index=False)
    pd.DataFrame(pathway_rows).to_csv(out / "B3_pathway_auroc.csv", index=False)
    (out / "B1_B2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
