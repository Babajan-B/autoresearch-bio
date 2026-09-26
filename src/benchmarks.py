from __future__ import annotations

from typing import Any


BENCHMARK_PRESETS: dict[str, dict[str, Any]] = {
    "gse96058_er": {
        "description": "Large breast cancer GEO RNA-seq benchmark using ER status on primary tumors only.",
        "input_mode": "geo",
        "accession": "GSE96058",
        "label_column": "er status",
        "dataset_id": "GSE96058_ER_primary",
        "normalization": "none",
        "phenotype_filters": {"er status": ["0", "1"]},
        "exclude_title_suffixes": ["repl"],
        "supplementary_urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/suppl/"
            "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz"
        ],
    },
    "gse81538_pam50_external": {
        "description": "External companion SCAN-B cohort using PAM50 luminal-vs-basal proxy labels for external transportability checks.",
        "input_mode": "geo",
        "accession": "GSE81538",
        "label_column": "pam50 subtype",
        "dataset_id": "GSE81538_PAM50_proxy_external",
        "normalization": "none",
        "phenotype_filters": {"pam50 subtype": ["Basal", "LumA", "LumB"]},
        "label_mapping": {"Basal": "0", "LumA": "1", "LumB": "1"},
        "supplementary_urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81538/suppl/GSE81538_gene_expression_405_transformed.csv.gz"
        ],
    },
}


def resolve_benchmark_preset(name: str) -> dict[str, Any]:
    key = name.strip().lower()
    if key not in BENCHMARK_PRESETS:
        available = ", ".join(sorted(BENCHMARK_PRESETS))
        raise ValueError(f"Unknown benchmark preset {name!r}. Available presets: {available}")
    return dict(BENCHMARK_PRESETS[key])
