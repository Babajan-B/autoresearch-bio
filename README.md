# LockLoop-Bio

A locked research loop for reproducible bulk RNA-seq phenotype classification.

LockLoop-Bio adapts the fixed-harness design of Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) to transcriptomic
machine learning. Decisions that usually stay open during an analysis are fixed
in code before any model is trained:

| Layer | Module | What is fixed |
|---|---|---|
| Locked preparation | `src/prepare.py` | cohort filters, label mapping, one stratified train/validation/test split (written once) |
| Iterative training | `src/train.py`, `src/models.py`, `src/features.py`, `src/ablations.py` | the only layer allowed to vary: model family, seed, number of genes |
| Locked evaluation | `src/evaluate.py`, `src/metrics.py`, `src/logging_utils.py` | threshold chosen on validation only; AUROC, Macro-F1, Brier, MCC; promotion by validation AUROC; timestamped run log |
| External check | `src/external_validation.py` | frozen models applied to a companion cohort, outside the audit log |

Guardrails enforced by the code:

- a logistic-regression baseline must be logged before any other model;
- genes are ranked by variance on the training split only;
- a run is accepted only if its validation AUROC beats the current best
  (ties: Macro-F1, −Brier, MCC); test metrics are calculated and logged but not used by the rule;
- every run adds a timestamped record (configuration, metrics, artefact paths and a git-commit field) to `results_log.json`; new runs are appended, and the only later change is relabelling a previously accepted run as superseded.

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

## Running the GEO benchmark

Data are downloaded from NCBI GEO and are not stored in this repository.
All run outputs are written to `reports/`.

```bash
# 1. Prepare the primary cohort (GSE96058, ER status, replicates excluded)
python -m src.cli prepare --benchmark-preset gse96058_er --input-mode geo \
  --label-column "er status" --output-dir data/processed/GSE96058_ER_primary

# 2. Required baseline, then any number of training runs and gene ablations
python -m src.cli baseline --prepared-dir data/processed/GSE96058_ER_primary \
  --output-dir reports/baseline/GSE96058_ER_primary
python -m src.cli train --prepared-dir data/processed/GSE96058_ER_primary \
  --model-family extra_trees --run-name <run_name> --output-dir reports/runs/<run_name> \
  --top-k-genes 2000 --random-seed 42
python -m src.cli ablate --prepared-dir data/processed/GSE96058_ER_primary \
  --genes ESR1,PGR,ERBB2 --output-dir reports/runs/<ablation_name>

# 3. Companion cohort (GSE81538) and external validation of a frozen run
python -m src.cli prepare --benchmark-preset gse81538_pam50_external --input-mode geo \
  --label-column "pam50 subtype" --output-dir data/processed/GSE81538_PAM50_proxy_external
python -m src.cli external-validate \
  --reference-run-dir reports/runs/<run_name> \
  --reference-prepared-dir data/processed/GSE96058_ER_primary \
  --external-prepared-dir data/processed/GSE81538_PAM50_proxy_external \
  --output-dir reports/external_validation/<run_name>_to_gse81538 \
  --run-name <run_name>_to_gse81538
```

Replace `<run_name>` with a run name of your choice; for external validation, use the run accepted in `results_log.json`.

## Repository layout

```
src/              framework (CLI: python -m src.cli {prepare,baseline,train,ablate,report,external-validate})
src/data_adapters local files, GEO and recount3 loaders
config/           default settings, benchmark preset, model hyperparameters
scripts/          shell/Python drivers for the benchmark runs
tests/            pytest suite with small synthetic fixtures
```

## Citation

If you use this code, please cite the repository (see `CITATION.cff`):

Banaganapalli B. LockLoop-Bio. https://github.com/Babajan-B/autoresearch-bio

## Licence

MIT — see `LICENSE`.
