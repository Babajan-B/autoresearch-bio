# Autoresearch-Bio

A locked research loop for reproducible bulk RNA-seq phenotype classification.

Autoresearch-Bio adapts the fixed-harness design of Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) to transcriptomic
machine learning. Decisions that usually stay open during an analysis are fixed
in code before any model is trained:

| Layer | Module | What is fixed |
|---|---|---|
| Locked preparation | `src/prepare.py` | cohort filters, label mapping, one stratified train/validation/test split (written once) |
| Iterative training | `src/train.py`, `src/models.py`, `src/features.py`, `src/ablations.py` | the only layer allowed to vary: model family, seed, number of genes |
| Locked evaluation | `src/evaluate.py`, `src/metrics.py`, `src/logging_utils.py` | threshold chosen on validation only; AUROC, Macro-F1, Brier, MCC; promotion by validation AUROC; append-only audit log |
| External check | `src/external_validation.py` | frozen models applied to a companion cohort, outside the audit log |

Guardrails enforced by the code:

- a logistic-regression baseline must be logged before any other model;
- genes are ranked by variance on the training split only;
- a run is accepted only if its validation AUROC beats the current best
  (ties: Macro-F1, −Brier, MCC); test metrics are logged but never read by the rule;
- every run appends its configuration, git commit, metrics and artefact paths to `results_log.json`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest          # unit and smoke tests (synthetic fixtures, no download)
```

Python ≥ 3.10.

## Quick start (synthetic data)

```bash
make smoke      # prepare → baseline → train → ablate → report on tests/fixtures
```

## Reproducing the manuscript benchmark

Data are downloaded from NCBI GEO and are not stored in this repository.

```bash
# 1. Prepare the primary cohort (GSE96058, ER status, replicates excluded)
python -m src.cli prepare --benchmark-preset gse96058_er --input-mode geo \
  --label-column "er status" --output-dir data/processed/GSE96058_ER_primary

# 2. Required baseline, then training runs, seeds, gene counts and the two declared ablations
python -m src.cli baseline --prepared-dir data/processed/GSE96058_ER_primary \
  --output-dir reports/baseline/GSE96058_ER_primary
bash scripts/run_paper_suite.sh data/processed/GSE96058_ER_primary

# 3. Companion cohort (GSE81538) and external validation of the frozen models
python -m src.cli prepare --benchmark-preset gse81538_pam50_external --input-mode geo \
  --label-column "pam50 subtype" --output-dir data/processed/GSE81538_PAM50_proxy_external
python -m src.cli external-validate \
  --reference-run-dir reports/runs/extra_trees_seed_7 \
  --reference-prepared-dir data/processed/GSE96058_ER_primary \
  --external-prepared-dir data/processed/GSE81538_PAM50_proxy_external \
  --output-dir reports/external_validation/extra_trees_seed_7_to_gse81538 \
  --run-name extra_trees_seed_7_to_gse81538

# 4. Statistics, tables and figures reported in the paper
pip install -e ".[paper]"
make paper      # writes to reports/paper/
```

`paper/` recomputes all confidence intervals (DeLong, stratified bootstrap),
paired model comparisons, calibration, the pathology-based external endpoint,
Table 1 and every figure from the saved per-run predictions; it never refits a model.

## Repository layout

```
src/              framework (CLI: python -m src.cli {prepare,baseline,train,ablate,report,external-validate})
src/data_adapters local files, GEO and recount3 loaders
config/           default settings, benchmark preset, model hyperparameters
scripts/          shell/Python drivers for the benchmark runs
paper/            analysis and figure scripts for the manuscript
tests/            pytest suite with small synthetic fixtures
```

## Citation

If you use this code, please cite:

Banaganapalli B, Khan S, Thalib HI, Shaik NA. Autoresearch-Bio: a locked research
loop for reproducible bulk RNA-seq phenotype classification. *Network Modeling
Analysis in Health Informatics and Bioinformatics* (under review).

## Licence

MIT — see `LICENSE`.
