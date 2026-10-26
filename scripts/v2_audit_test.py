"""Audit-trail test (Methods 2.9): verify the study log, then tamper with copies and record what verify-log reports.

Usage: python -m scripts.v2_audit_test [--out results/v2/audit_test.json]
The log stores relative paths, so results/v2 and config/v2 are copied into a temporary directory and every change is
made there; the real files are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from src.logging_utils import entry_digest, verify_log

LOG = "results/v2/study_log.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/v2/audit_test.json")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    results = {"original": verify_log(LOG)}
    root = Path.cwd()

    def scenario(name, mutate):
        with tempfile.TemporaryDirectory() as tmp:
            for d in ("results/v2", "config/v2"):
                shutil.copytree(root / d, Path(tmp) / d, ignore=shutil.ignore_patterns("*.tif", "*.png", "*.pdf"))
            os.chdir(tmp)
            try:
                mutate(Path(tmp))
                results[name] = verify_log(LOG)
            finally:
                os.chdir(root)

    def edit_log(fn):
        def mutate(tmp):
            entries = json.loads((tmp / LOG).read_text())
            fn(entries)
            (tmp / LOG).write_text(json.dumps(entries))
        return mutate

    def edited_value(entries):
        entries[0]["repeats"] = 19

    def edited_and_rehashed(entries):
        edited_value(entries)
        entries[0]["entry_hash"] = entry_digest(entries[0])

    def entry_deleted(entries):
        del entries[0]

    def declaration_edited(tmp):
        d = tmp / "config/v2/DECLARATION.md"
        d.write_text(d.read_text() + "\nedited after the runs\n")

    def results_file_edited(tmp):
        first = json.loads((tmp / LOG).read_text())[0]
        f = tmp / first["outputs"]["picks"]
        f.write_text(f.read_text().replace("0.9", "0.8", 1))

    scenario("edited_value", edit_log(edited_value))
    scenario("edited_and_rehashed", edit_log(edited_and_rehashed))
    scenario("first_entry_deleted", edit_log(entry_deleted))
    scenario("declaration_edited", declaration_edited)
    scenario("results_file_edited", results_file_edited)
    summary = {k: {"detected": bool(v), "messages": v} for k, v in results.items()}
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v["detected"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
