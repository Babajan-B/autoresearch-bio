"""Per-sample pathway (gene-set) scores.

Each sample is scored on its own: genes are ranked within the sample and a set's score is the mean rank of its
genes, scaled to [-0.5, 0.5] (0 = no different from a random gene set). Because nothing is estimated across
samples, the scores need no training data and carry no information from one cohort to another.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_gmt(path: str | Path) -> dict[str, list[str]]:
    sets = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) > 2:
            sets[fields[0].strip()] = [g for g in fields[2:] if g]
    return sets


def rank_scores(expression: pd.DataFrame, gene_sets: dict[str, list[str]], min_genes: int = 10) -> pd.DataFrame:
    """samples x genes -> samples x gene sets."""
    ranks = expression.rank(axis=1, method="average").to_numpy() / expression.shape[1]  # in (0, 1]
    columns = {name: i for i, name in enumerate(expression.columns)}
    scores = {}
    for name, genes in gene_sets.items():
        idx = [columns[g] for g in genes if g in columns]
        if len(idx) >= min_genes:
            scores[name] = ranks[:, idx].mean(axis=1) - 0.5
    return pd.DataFrame(scores, index=expression.index).astype(np.float32)
