"""Run the declared v2 selection study for some or all endpoints.

Usage: python -m scripts.v2_run_study --data-root ../.. [--endpoints er her2] [--repeats 20] [--allow-dirty]
Refuses to start from uncommitted code or from data whose hashes differ from config/v2/study.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.logging_utils import append_result
from src.provenance import collect_provenance
from src.selection_study import policies_from_config, run_endpoint, write_outputs
from src.utils.io import sha256_file, utc_now

CONFIG = Path("config/v2/study.yaml")
DECLARATION = Path("config/v2/DECLARATION.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".", help="directory that contains data/v2")
    parser.add_argument("--endpoints", nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=None, help="override for smoke runs only")
    parser.add_argument("--out", default="results/v2")
    parser.add_argument("--allow-dirty", action="store_true", help="for smoke runs only; such runs are flagged")
    args = parser.parse_args()

    study = yaml.safe_load(CONFIG.read_text())
    root = Path(args.data_root)
    for name, expected in study["data"]["sha256"].items():
        actual = sha256_file(root / "data/v2" / name)
        if actual != expected:
            raise SystemExit(f"{name}: hash {actual} differs from the declared {expected}")
    provenance = collect_provenance(root / "data/v2", DECLARATION, require_clean=not args.allow_dirty)
    provenance.update(study_config_path=str(CONFIG), study_config_sha256=sha256_file(CONFIG))

    scanb = pd.read_pickle(root / study["data"]["scanb_expression"])
    scanb_labels = pd.read_csv(root / study["data"]["scanb_labels"], index_col=0)
    tcga = pd.read_pickle(root / study["data"]["tcga_expression"])
    tcga_labels = pd.read_csv(root / study["data"]["tcga_labels"], index_col=0)
    design, policies = study["design"], policies_from_config(study["policies"])
    repeats = args.repeats or design["repeats"]

    for endpoint in args.endpoints or study["endpoints"]:
        started = utc_now()
        external = (tcga, tcga_labels[endpoint]) if endpoint in study["external_endpoints"] else None
        scores, picks = run_endpoint(
            scanb, scanb_labels[endpoint], grid=study["grid"], policies=policies, baseline=study["baseline"],
            repeats=repeats, folds=design["cv_folds"], vault_fraction=design["vault_fraction"],
            test_fraction=design["test_fraction"], vault_seed=design["vault_seed"],
            repeat_seed=design["repeat_seed"], external=external,
            progress=lambda message: print(f"[{endpoint}] {message}", flush=True))
        paths = write_outputs(Path(args.out) / endpoint, scores, picks)
        append_result({
            "timestamp": utc_now(), "started": started, "dataset_id": f"selection_study_v2/{endpoint}",
            "run_name": f"selection_study_{endpoint}", "run_type": "study", "status": "complete",
            "repeats": repeats, "smoke": bool(args.allow_dirty or args.repeats),
            "outputs": {k: str(p) for k, p in paths.items()},
            "outputs_sha256": {k: sha256_file(p) for k, p in paths.items()},
            "provenance": provenance,
        }, Path(args.out) / "study_log.json")
        print(f"[{endpoint}] written to {paths['picks'].parent}", flush=True)


if __name__ == "__main__":
    main()
