"""Build the harmonised SCAN-B (GSE96058) and TCGA-BRCA (recount3) matrices and endpoint labels for the v2 study.

Both cohorts are put on the same scale with sample-wise operations only (no cohort-level batch correction, so
nothing learned from one cohort reaches the other): restrict to protein-coding genes present in both, convert to
TPM within each sample over that shared gene set, then log2(TPM + 1).

Inputs (downloaded by scripts/v2_download.sh) are read from data/raw; outputs go to data/v2.
Usage: python scripts/v2_prepare_cohorts.py [--raw data/raw] [--out data/v2]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import sha256_file

SCANB_ENDPOINTS = {
    # endpoint: (column, negative values, positive values)
    "er": ("er status", {"0"}, {"1"}),
    "pgr": ("pgr status", {"0"}, {"1"}),
    "her2": ("her2 status", {"0"}, {"1"}),
    "ki67": ("ki67 status", {"0"}, {"1"}),
    "nhg3": ("nhg", {"G1", "G2"}, {"G3"}),
    "luma_lumb": ("pam50 subtype", {"LumA"}, {"LumB"}),
    "node": ("lymph node status", {"NodeNegative"}, {"NodePositive"}),
}


def _binary(series: pd.Series, negative: set[str], positive: set[str]) -> pd.Series:
    values = series.astype(str).str.strip()
    return pd.Series(np.where(values.isin(positive), 1.0, np.where(values.isin(negative), 0.0, np.nan)), index=series.index)


def read_series_matrix(path: Path) -> pd.DataFrame:
    titles, characteristics = None, []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!Sample_title"):
                titles = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                characteristics.append([v.strip('"') for v in line.rstrip("\n").split("\t")[1:]])
    columns = {}
    for row in characteristics:
        key = next((v.split(": ", 1)[0] for v in row if ": " in v), None)
        if key:
            columns[key] = [v.split(": ", 1)[1] if ": " in v else np.nan for v in row]
    return pd.DataFrame(columns, index=titles)


def protein_coding_lengths(gtf_path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(gtf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[2] != "gene" or 'gene_type "protein_coding"' not in fields[8]:
                continue
            # recount3 stores the summed exon length (bp) in the GTF score column
            rows.append((re.search(r'gene_id "([^"]+)"', fields[8]).group(1),
                         re.search(r'gene_name "([^"]+)"', fields[8]).group(1), float(fields[5])))
    return pd.DataFrame(rows, columns=["gene_id", "symbol", "length"]).set_index("gene_id")


def to_log_tpm(per_length: pd.DataFrame) -> pd.DataFrame:
    """samples x genes, values proportional to reads per base -> log2(TPM + 1) within each sample."""
    tpm = per_length.div(per_length.sum(axis=1), axis=0) * 1e6
    return np.log2(tpm + 1).astype(np.float32)


def tcga_labels(meta: pd.DataFrame) -> pd.DataFrame:
    er = _binary(meta["xml_breast_carcinoma_estrogen_receptor_status"], {"Negative"}, {"Positive"})
    pgr = _binary(meta["xml_breast_carcinoma_progesterone_receptor_status"], {"Negative"}, {"Positive"})
    ihc = meta["xml_lab_proc_her2_neu_immunohistochemistry_receptor_status"].astype(str)
    ish = meta["xml_lab_procedure_her2_neu_in_situ_hybrid_outcome_type"].astype(str)
    # ASCO/CAP logic: ISH decides when done; otherwise IHC positive/negative; equivocal without ISH is excluded
    her2 = pd.Series(np.nan, index=meta.index)
    her2[ish.eq("Positive")] = 1.0
    her2[ish.eq("Negative")] = 0.0
    no_ish = ~ish.isin(["Positive", "Negative"])
    her2[no_ish & ihc.eq("Positive")] = 1.0
    her2[no_ish & ihc.eq("Negative")] = 0.0
    n_stage = meta["cgc_case_pathologic_n"].astype(str)
    node = pd.Series(np.where(n_stage.str.match(r"^N0"), 0.0, np.where(n_stage.str.match(r"^N[1-3]"), 1.0, np.nan)), index=meta.index)
    return pd.DataFrame({"er": er, "pgr": pgr, "her2": her2, "node": node})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/v2")
    args = parser.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    genes = protein_coding_lengths(raw / "tcga_brca/human.gene_sums.G026.gtf.gz")

    # ---- SCAN-B: log2(FPKM + 0.1), genes x samples, sample titles as columns
    scanb = pd.read_csv(raw / "gse96058/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz", index_col=0)
    scanb = scanb.loc[:, ~scanb.columns.str.endswith("repl")]
    shared = sorted(set(scanb.index) & set(genes["symbol"]))
    scanb_fpkm = (2.0 ** scanb.loc[shared].T - 0.1).clip(lower=0)  # back to FPKM (per-length already)

    # ---- TCGA: base-pair coverage sums, genes x samples
    counts = pd.read_csv(raw / "tcga_brca/tcga.gene_sums.BRCA.G026.gz", sep="\t", comment="#", index_col=0)
    counts = counts.loc[counts.index.intersection(genes.index)]
    per_length = counts.div(genes.loc[counts.index, "length"], axis=0)
    per_length = per_length.groupby(genes.loc[per_length.index, "symbol"]).sum().loc[shared].T

    meta = pd.read_csv(raw / "tcga_brca/tcga.tcga.BRCA.MD.gz", sep="\t", low_memory=False).set_index("external_id")
    meta = meta.loc[meta.index.intersection(per_length.index)]
    meta = meta[meta["cgc_sample_sample_type_code"] == 1]  # primary tumours
    meta = meta.assign(patient=meta["tcga_barcode"].str[:12]).sort_values("tcga_barcode")
    meta = meta[~meta["patient"].duplicated()]  # one sample per patient (first barcode)

    scanb_expr = to_log_tpm(scanb_fpkm)
    tcga_expr = to_log_tpm(per_length.loc[meta.index])
    tcga_expr.index = meta.loc[tcga_expr.index, "patient"]

    pheno = pd.concat([read_series_matrix(p) for p in sorted((raw / "gse96058").glob("GSE96058-GPL*_series_matrix.txt.gz"))])
    pheno = pheno.loc[pheno.index.intersection(scanb_expr.index)]
    scanb_lab = pd.DataFrame({name: _binary(pheno[col], neg, pos) for name, (col, neg, pos) in SCANB_ENDPOINTS.items()})
    tcga_lab = tcga_labels(meta).set_axis(meta["patient"].values)

    scanb_expr.to_pickle(out / "scanb_expr.pkl")
    tcga_expr.to_pickle(out / "tcga_expr.pkl")
    scanb_lab.to_csv(out / "scanb_labels.csv")
    tcga_lab.to_csv(out / "tcga_labels.csv")
    pheno.to_csv(out / "scanb_phenotype.csv")

    summary = {
        "genes": len(shared),
        "scanb_samples": int(scanb_expr.shape[0]),
        "tcga_samples": int(tcga_expr.shape[0]),
        "scanb_endpoints": {c: {"n": int(scanb_lab[c].notna().sum()), "positive": int(scanb_lab[c].sum())} for c in scanb_lab},
        "tcga_endpoints": {c: {"n": int(tcga_lab[c].notna().sum()), "positive": int(tcga_lab[c].sum())} for c in tcga_lab},
        "sha256": {p.name: sha256_file(p) for p in sorted(out.glob("*")) if p.suffix in {".pkl", ".csv"}},
    }
    (out / "cohort_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
