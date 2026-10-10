from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.benchmarks import BENCHMARK_PRESETS, resolve_benchmark_preset
from src.ablations import run_gene_ablation
from src.external_validation import run_external_validation
from src.logging_utils import read_results_log, render_markdown_report
from src.prepare import prepare_dataset, prepared_summary
from src.train import run_baseline, run_training


def _parse_genes(value: str) -> list[str]:
    return [gene.strip() for gene in value.split(",") if gene.strip()]


def _parse_label_map(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--label-map must decode to a JSON object")
    return {str(key): str(mapping) for key, mapping in payload.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LockLoop-Bio terminal workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare and freeze a dataset")
    prepare_parser.add_argument("--benchmark-preset", choices=sorted(BENCHMARK_PRESETS), default=None)
    prepare_parser.add_argument("--input-mode", choices=["local", "geo", "recount3"], required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--label-column", required=True)
    prepare_parser.add_argument("--normalization", default="auto", choices=["auto", "none", "log1p"])
    prepare_parser.add_argument("--random-seed", type=int, default=42)
    prepare_parser.add_argument("--val-fraction", type=float, default=0.2)
    prepare_parser.add_argument("--test-fraction", type=float, default=0.2)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument("--expression-path")
    prepare_parser.add_argument("--phenotype-path")
    prepare_parser.add_argument("--sample-id-column", default="sample_id")
    prepare_parser.add_argument("--dataset-id")
    prepare_parser.add_argument("--accession")
    prepare_parser.add_argument("--project")
    prepare_parser.add_argument("--organism", default="human")
    prepare_parser.add_argument("--project-home", default="data_sources/sra")
    prepare_parser.add_argument("--annotation", default="G026")
    prepare_parser.add_argument("--label-map", default=None, help="JSON object mapping raw label values to normalized labels")

    baseline_parser = subparsers.add_parser("baseline", help="Run the required logistic-regression baseline")
    baseline_parser.add_argument("--prepared-dir", required=True)
    baseline_parser.add_argument("--output-dir", required=True)
    baseline_parser.add_argument("--top-k-genes", type=int, default=1000)
    baseline_parser.add_argument("--log-path", default="results_log.json")
    baseline_parser.add_argument("--random-seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="Run non-baseline model training")
    train_parser.add_argument("--prepared-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--model-family", choices=["logistic_regression", "random_forest", "extra_trees", "hist_gradient_boosting"], required=True)
    train_parser.add_argument("--run-name", required=True)
    train_parser.add_argument("--top-k-genes", type=int, default=2000)
    train_parser.add_argument("--log-path", default="results_log.json")
    train_parser.add_argument("--random-seed", type=int, default=42)
    train_parser.add_argument("--notes", default="")

    ablate_parser = subparsers.add_parser("ablate", help="Run shortcut-gene ablation")
    ablate_parser.add_argument("--prepared-dir", required=True)
    ablate_parser.add_argument("--output-dir", required=True)
    ablate_parser.add_argument("--genes", default="ESR1,PGR,ERBB2")
    ablate_parser.add_argument("--model-family", default="logistic_regression")
    ablate_parser.add_argument("--top-k-genes", type=int, default=1000)
    ablate_parser.add_argument("--log-path", default="results_log.json")
    ablate_parser.add_argument("--random-seed", type=int, default=42)

    report_parser = subparsers.add_parser("report", help="Generate a Markdown report from results_log.json")
    report_parser.add_argument("--log-path", default="results_log.json")
    report_parser.add_argument("--output-path", required=True)

    external_parser = subparsers.add_parser("external-validate", help="Evaluate a trained model on an external prepared cohort")
    external_parser.add_argument("--reference-run-dir", required=True)
    external_parser.add_argument("--reference-prepared-dir", required=True)
    external_parser.add_argument("--external-prepared-dir", required=True)
    external_parser.add_argument("--output-dir", required=True)
    external_parser.add_argument("--run-name", required=True)
    external_parser.add_argument("--notes", default="")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        input_mode = args.input_mode
        label_column = args.label_column
        normalization = args.normalization
        phenotype_filters = None
        exclude_title_suffixes = None
        label_mapping = _parse_label_map(args.label_map)
        kwargs = {}
        if args.benchmark_preset:
            preset = resolve_benchmark_preset(args.benchmark_preset)
            input_mode = preset["input_mode"]
            label_column = preset["label_column"]
            normalization = preset.get("normalization", normalization)
            phenotype_filters = preset.get("phenotype_filters")
            exclude_title_suffixes = preset.get("exclude_title_suffixes")
            label_mapping = preset.get("label_mapping", label_mapping)
            kwargs.update(
                {
                    key: value
                    for key, value in preset.items()
                    if key in {"accession", "project", "organism", "project_home", "annotation", "dataset_id", "supplementary_urls"}
                }
            )

        if input_mode == "local":
            if not args.expression_path or not args.phenotype_path:
                parser.error("local prepare mode requires --expression-path and --phenotype-path")
            kwargs.update(
                {
                    "expression_path": args.expression_path,
                    "phenotype_path": args.phenotype_path,
                    "sample_id_column": args.sample_id_column,
                    "dataset_id": args.dataset_id,
                }
            )
        elif input_mode == "geo":
            accession = kwargs.get("accession") or args.accession
            if not accession:
                parser.error("geo prepare mode requires --accession")
            kwargs["accession"] = accession
        else:
            project = kwargs.get("project") or args.project
            if not project:
                parser.error("recount3 prepare mode requires --project")
            kwargs.update(
                {
                    "project": project,
                    "organism": kwargs.get("organism", args.organism),
                    "project_home": kwargs.get("project_home", args.project_home),
                    "annotation": kwargs.get("annotation", args.annotation),
                }
            )

        prepared = prepare_dataset(
            input_mode=input_mode,
            output_dir=args.output_dir,
            label_column=label_column,
            normalization=normalization,
            random_seed=args.random_seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            overwrite=args.overwrite,
            phenotype_filters=phenotype_filters,
            exclude_title_suffixes=exclude_title_suffixes,
            label_mapping=label_mapping,
            **kwargs,
        )
        print(json.dumps(prepared_summary(prepared), indent=2))
        return

    if args.command == "baseline":
        summary = run_baseline(
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            top_k_genes=args.top_k_genes,
            log_path=args.log_path,
            random_seed=args.random_seed,
        )
        print(json.dumps(asdict(summary), indent=2))
        return

    if args.command == "train":
        summary = run_training(
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            model_family=args.model_family,
            top_k_genes=args.top_k_genes,
            run_name=args.run_name,
            log_path=args.log_path,
            random_seed=args.random_seed,
            notes=args.notes,
        )
        print(json.dumps(asdict(summary), indent=2))
        return

    if args.command == "ablate":
        summary = run_gene_ablation(
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            genes=_parse_genes(args.genes),
            model_family=args.model_family,
            top_k_genes=args.top_k_genes,
            log_path=args.log_path,
            random_seed=args.random_seed,
        )
        print(json.dumps(asdict(summary), indent=2))
        return

    if args.command == "report":
        entries = read_results_log(args.log_path)
        report = render_markdown_report(entries)
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(str(output_path))
        return

    if args.command == "external-validate":
        summary = run_external_validation(
            reference_run_dir=args.reference_run_dir,
            reference_prepared_dir=args.reference_prepared_dir,
            external_prepared_dir=args.external_prepared_dir,
            output_dir=args.output_dir,
            run_name=args.run_name,
            notes=args.notes,
        )
        print(json.dumps(asdict(summary), indent=2))
        return


if __name__ == "__main__":
    main()
