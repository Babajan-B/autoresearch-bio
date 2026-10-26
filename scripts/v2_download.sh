#!/usr/bin/env bash
# Raw inputs for the v2 study. Run from the directory that should hold data/raw.
set -euo pipefail
mkdir -p data/raw/gse96058 data/raw/tcga_brca
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058
R3=https://duffel.rail.bio/recount3/human
( cd data/raw/gse96058
  curl -fsSLO $GEO/suppl/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz
  curl -fsSLO $GEO/matrix/GSE96058-GPL11154_series_matrix.txt.gz
  curl -fsSLO $GEO/matrix/GSE96058-GPL18573_series_matrix.txt.gz )
( cd data/raw/tcga_brca
  curl -fsSLO $R3/data_sources/tcga/gene_sums/CA/BRCA/tcga.gene_sums.BRCA.G026.gz
  curl -fsSLO $R3/data_sources/tcga/metadata/CA/BRCA/tcga.tcga.BRCA.MD.gz
  curl -fsSLO $R3/annotations/gene_sums/human.gene_sums.G026.gtf.gz )
