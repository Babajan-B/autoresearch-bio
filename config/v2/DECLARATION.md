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
