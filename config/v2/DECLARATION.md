# Declaration: selection study v2

Written and committed before any v2 model was trained. The commit that adds this file, and its SHA-256 stored in
every log entry, fix the plan below. Anything not listed here and reported later is labelled post hoc.

## Question
When many reasonable candidate models are fitted to a transcriptomic endpoint, how much does the rule used to
choose the reported model change (a) the performance that gets reported and (b) the performance that holds up on
data no strategy saw, in the same cohort and in an independent cohort?

## Data
- SCAN-B GSE96058 primary tumours (technical replicates removed), 3,273 tumours.
- TCGA-BRCA (recount3, G026), primary tumours, one sample per patient, 1,095 patients.
- Both reduced to 18,564 shared protein-coding genes and converted within each sample to log2(TPM + 1)
  (scripts/v2_prepare_cohorts.py). No cohort-level batch correction. File hashes are in study.yaml.

## Endpoints (binary; missing or other categories excluded)
| Endpoint | SCAN-B definition | TCGA definition |
|---|---|---|
| er | ER status 0/1 | ER IHC Negative/Positive |
| pgr | PgR status 0/1 | PgR IHC Negative/Positive |
| her2 | HER2 status 0/1 | ISH result if done, else IHC Positive/Negative; equivocal without ISH excluded |
| node | NodeNegative/NodePositive | pathologic N0* vs N1–N3* |
| nhg3 | NHG G3 vs G1–G2 | — |
| ki67 | Ki67 status 0/1 | — |
| luma_lumb | PAM50 LumB vs LumA | — |

## Design
- Sealed vault: 25% of each endpoint's SCAN-B tumours, stratified, seed 2026; never used for any choice.
- 20 repeats: remaining tumours split 80/20 into development and test (seed 1000 + r).
- 12 candidates (4 model families with the framework's fixed hyperparameters × 500/2,000/5,000 genes by
  training-sample variance). Baseline: logistic regression, 2,000 genes.
- Every candidate: 5-fold stratified CV on development (genes re-selected inside each fold); the fold-1 model is
  the single-split model of the legacy rule; refit on all development data for the CV-based policies.
- Strategies: post hoc best test AUROC; the seven policies in study.yaml; baseline only.

## Primary outcomes
1. Optimism: reported test AUROC minus vault AUROC of the chosen model, per strategy (mean over repeats).
2. External transfer (er, pgr, her2, node): TCGA AUROC, Brier score and calibration slope of the chosen model.
3. Regret: best candidate's vault (or TCGA) AUROC minus the chosen model's.
4. Stability: how often each strategy picks the same candidate across repeats.

## Secondary
Average precision for the minority class; differences between endpoints of strong (er, her2, pgr) and weak
(node, ki67, luma_lumb, nhg3) transcriptomic signal.

## Not planned (anything of this kind will be labelled post hoc)
Hyperparameter tuning, batch correction, recalibration, additional cohorts or endpoints.

## Amendment 1 (2026-10-26, before any v2 result was seen)

The first launch was stopped after about ten minutes. No output file had been written and no result was looked at
(results/v2/aborted_v2.0 holds the two progress logs). Reasons: gradient boosting with 2,000 and 5,000 genes made
the run about 14 hours, and the study had no candidate built on biological prior knowledge. Changes:

1. Gradient boosting is kept at 500 genes only; the 2,000- and 5,000-gene variants are removed.
2. Two pathway-level candidates are added: logistic regression and random forest on 50 MSigDB Hallmark scores,
   each computed within a sample as the mean within-sample rank of the set's genes (src/modules.py). These scores
   use no training data and need no cross-cohort adjustment. Gene-set file hash in study.yaml.
3. The calibration-in-the-large (mean predicted probability minus observed prevalence) is logged for every hold-out.
4. The grid therefore has 12 candidates: logistic regression, random forest and Extra Trees × 500/2,000/5,000
   genes; gradient boosting × 500 genes; two Hallmark candidates. Everything else is unchanged.

### Biological analyses (declared now; scripts/v2_biology.py)
All use the non-vault SCAN-B tumours and TCGA-BRCA only; the vault stays sealed.
- B1 Signal strength: per-gene AUROC for each endpoint; number of genes with AUROC ≥ 0.70 or ≤ 0.30; top genes.
- B2 Label-blind gene filtering versus signal: share of the 100 most informative genes retained by the variance
  filter at 500/2,000/5,000 genes, and the variance rank of marker genes (ESR1, PGR, FOXA1, GATA3, ERBB2, GRB7,
  STARD3, PGAP3, MIEN1, MKI67, TOP2A, AURKA).
- B3 Pathway programmes: AUROC of each Hallmark score per endpoint in SCAN-B and in TCGA, and their agreement
  between cohorts (Spearman correlation).
- B4 Cohort shift: endpoint prevalence in each cohort, related to the TCGA calibration-in-the-large of the models
  chosen by each strategy.
- B5 Optimism of post hoc selection against endpoint signal strength (descriptive, seven endpoints).
