PYTHON ?= python3
CLI = $(PYTHON) -m src.cli

.PHONY: install test smoke baseline paper clean

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	pytest

baseline:
	$(CLI) prepare --input-mode local --expression-path tests/fixtures/local_expression.tsv --phenotype-path tests/fixtures/local_phenotype.tsv --label-column label --output-dir data/processed/local_smoke
	$(CLI) baseline --prepared-dir data/processed/local_smoke --output-dir reports/baseline/local_smoke

smoke: baseline
	$(CLI) train --prepared-dir data/processed/local_smoke --model-family random_forest --run-name rf_smoke --output-dir reports/runs/rf_smoke
	$(CLI) ablate --prepared-dir data/processed/local_smoke --genes ESR1,PGR,ERBB2 --output-dir reports/runs/shortcut_ablation
	$(CLI) report --output-path reports/final/smoke_report.md

paper:
	bash paper/run_all.sh

clean:
	rm -rf .pytest_cache reports/baseline/local_smoke reports/runs/rf_smoke reports/runs/shortcut_ablation reports/final/smoke_report.md
