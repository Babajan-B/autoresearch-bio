from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.utils.io import read_json, sha256_file, write_json


def read_results_log(log_path: str | Path = "results_log.json") -> list[dict[str, Any]]:
    payload = read_json(log_path, default=[])
    if not isinstance(payload, list):
        raise ValueError("results_log.json must contain a JSON array")
    return payload


def metric_rank_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("auroc", 0.0)),
        float(metrics.get("macro_f1", 0.0)),
        -float(metrics.get("brier_score", 1.0)),
        float(metrics.get("mcc", 0.0)),
    )


def baseline_exists(log_path: str | Path, dataset_id: str) -> bool:
    return any(
        entry.get("dataset_id") == dataset_id and entry.get("run_type") == "baseline"
        for entry in read_results_log(log_path)
    )


def _as_candidate(entry: dict[str, Any], source: str) -> dict[str, Any]:
    cv = source == "cv"
    return {
        "model_family": entry.get("model_family"),
        "feature_count": entry.get("feature_count", 0),
        "is_baseline": entry.get("run_type") == "baseline",
        "metrics": entry["cv_metrics"] if cv else entry["validation_metrics"],
        "se": entry.get("cv_se", {}) if cv else {},
    }


def determine_acceptance(
    log_path: str | Path,
    dataset_id: str,
    validation_metrics: dict[str, Any],
    run_type: str,
    policy: Any = None,
    candidate_entry: dict[str, Any] | None = None,
) -> str:
    if run_type == "baseline":
        return "accepted"

    if policy is not None:
        # set-based choice over the baseline and all training runs; ablations are diagnostics and never promoted
        from src.promotion import choose

        if run_type != "train":
            return "rejected"
        logged = [e for e in read_results_log(log_path)
                  if e.get("dataset_id") == dataset_id and e.get("run_type") in {"baseline", "train"}]
        pool = [_as_candidate(e, policy.source) for e in logged] + [_as_candidate(candidate_entry, policy.source)]
        return "accepted" if choose(pool, policy) == len(pool) - 1 else "rejected"

    previous_entries = [
        entry
        for entry in read_results_log(log_path)
        if entry.get("dataset_id") == dataset_id and entry.get("status") == "accepted"
    ]
    if not previous_entries:
        return "accepted"

    previous_best = max(previous_entries, key=lambda entry: metric_rank_tuple(entry["validation_metrics"]))
    if metric_rank_tuple(validation_metrics) > metric_rank_tuple(previous_best["validation_metrics"]):
        return "accepted"
    return "rejected"


GENESIS_HASH = "0" * 64


def entry_digest(entry: dict[str, Any]) -> str:
    """SHA-256 of an entry's canonical JSON, excluding its own entry_hash field."""
    body = {key: value for key, value in entry.items() if key != "entry_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def append_result(entry: dict[str, Any], log_path: str | Path = "results_log.json") -> dict[str, Any]:
    """Append one entry to the hash-chained log. Earlier entries are never rewritten."""
    # shortcut: fcntl lock is POSIX-only; add a Windows lock if the framework is ever run there in parallel
    import fcntl

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{log_path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _append_locked(entry, log_path)


def _append_locked(entry: dict[str, Any], log_path: str | Path) -> dict[str, Any]:
    log_entries = read_results_log(log_path)
    # round-trip through JSON so the hash is computed on exactly what is stored
    record = json.loads(json.dumps(entry, sort_keys=True))
    record["prev_hash"] = log_entries[-1].get("entry_hash", GENESIS_HASH) if log_entries else GENESIS_HASH
    record["entry_hash"] = entry_digest(record)
    log_entries.append(record)
    write_json(log_path, log_entries)
    return record


def verify_log(log_path: str | Path = "results_log.json") -> list[str]:
    """Return a list of integrity problems; an empty list means the chain is intact."""
    problems: list[str] = []
    previous = GENESIS_HASH
    for index, entry in enumerate(read_results_log(log_path)):
        name = entry.get("run_name", "?")
        if "entry_hash" not in entry:
            problems.append(f"entry {index} ({name}): no entry_hash (written before hash chaining)")
            previous = GENESIS_HASH
            continue
        if entry.get("prev_hash") != previous:
            problems.append(f"entry {index} ({name}): prev_hash does not match the preceding entry")
        if entry_digest(entry) != entry["entry_hash"]:
            problems.append(f"entry {index} ({name}): content does not match entry_hash")
        for path_key, hash_key in (("declaration_path", "declaration_sha256"), ("splits_path", "splits_sha256")):
            path, expected = entry.get("provenance", {}).get(path_key), entry.get("provenance", {}).get(hash_key)
            if path and expected and Path(path).exists() and sha256_file(path) != expected:
                problems.append(f"entry {index} ({name}): {path} changed since the run")
        for key, path in entry.get("outputs", {}).items():
            expected = entry.get("outputs_sha256", {}).get(key)
            if expected and Path(path).exists() and sha256_file(path) != expected:
                problems.append(f"entry {index} ({name}): output {path} changed since the run")
        previous = entry["entry_hash"]
    return problems


def effective_status(log_entries: list[dict[str, Any]]) -> dict[int, str]:
    """Status of each entry as of now: accepted runs that a later accepted run replaced become superseded."""
    status = {index: entry.get("status", "") for index, entry in enumerate(log_entries)}
    last_accepted: dict[str, int] = {}
    for index, entry in enumerate(log_entries):
        if entry.get("status") == "accepted" and entry.get("run_type") != "baseline":
            dataset_id = entry.get("dataset_id")
            if dataset_id in last_accepted:
                status[last_accepted[dataset_id]] = "superseded"
            last_accepted[dataset_id] = index
    return status


def render_markdown_report(log_entries: list[dict[str, Any]]) -> str:
    if not log_entries:
        return "# LockLoop-Bio Report\n\nNo runs have been logged yet.\n"

    lines = ["# LockLoop-Bio Report", ""]
    status = effective_status(log_entries)
    datasets = sorted({entry.get("dataset_id", "unknown") for entry in log_entries})
    for dataset_id in datasets:
        indices = [index for index, entry in enumerate(log_entries) if entry.get("dataset_id") == dataset_id]
        accepted = [index for index in indices if status[index] == "accepted"]
        # the current model is the most recent accepted run (a non-baseline one if any was accepted)
        best_entry = log_entries[max(accepted, key=lambda i: (log_entries[i].get("run_type") != "baseline", i))] if accepted else log_entries[indices[-1]]

        lines.append(f"## {dataset_id}")
        lines.append("")
        lines.append(f"- Total runs: {len(indices)}")
        lines.append(f"- Accepted runs: {len(accepted)}")
        lines.append(f"- Best run: {best_entry.get('run_name')}")
        lines.append(f"- Best model family: {best_entry.get('model_family')}")
        lines.append(
            "- Validation metrics: "
            f"AUROC={best_entry['validation_metrics']['auroc']:.4f}, "
            f"Macro-F1={best_entry['validation_metrics']['macro_f1']:.4f}, "
            f"Brier={best_entry['validation_metrics']['brier_score']:.4f}, "
            f"MCC={best_entry['validation_metrics']['mcc']:.4f}"
        )
        lines.append(
            "- Test metrics: "
            f"AUROC={best_entry['test_metrics']['auroc']:.4f}, "
            f"Macro-F1={best_entry['test_metrics']['macro_f1']:.4f}, "
            f"Brier={best_entry['test_metrics']['brier_score']:.4f}, "
            f"MCC={best_entry['test_metrics']['mcc']:.4f}"
        )
        lines.append(f"- Output directory: {best_entry.get('output_dir')}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
