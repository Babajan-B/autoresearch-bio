from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class FeatureSelectionResult:
    selected_genes: list[str]
    excluded_genes: list[str]


def resolve_excluded_genes(columns: pd.Index, excluded_genes: list[str] | None) -> list[str]:
    if not excluded_genes:
        return []
    normalized = {gene.upper() for gene in excluded_genes}
    return [column for column in columns if str(column).upper() in normalized]


def select_top_variable_genes(
    train_expression: pd.DataFrame,
    top_k_genes: int,
    excluded_genes: list[str] | None = None,
) -> FeatureSelectionResult:
    to_exclude = resolve_excluded_genes(train_expression.columns, excluded_genes)
    filtered = train_expression.drop(columns=to_exclude, errors="ignore")
    if filtered.empty:
        raise ValueError("No genes remain after applying exclusions")

    variances = filtered.var(axis=0).sort_values(ascending=False)
    selected = variances.head(min(top_k_genes, len(variances))).index.tolist()
    if not selected:
        raise ValueError("Feature selection produced an empty gene set")
    return FeatureSelectionResult(selected_genes=selected, excluded_genes=to_exclude)
