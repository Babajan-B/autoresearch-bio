"""Declared promotion policies.

A policy states, before any model is trained, how the reported model will be chosen from the logged runs:
the primary metric and where it is measured (one validation split or cross-validation on the development set),
how large a difference counts (exact, an absolute margin, or one standard error), constraints a model must meet
(for example a calibration slope range), and the order of tie-breakers (for example simpler model first).
Choice is made over the whole set of logged candidates, so it does not depend on the order in which runs were made.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# lower rank = simpler; used by the "complexity" tie-breaker
COMPLEXITY = {"logistic_regression": 0, "random_forest": 1, "extra_trees": 1, "hist_gradient_boosting": 2}
LOWER_IS_BETTER = {"brier_score", "log_loss", "abs_log_calibration_slope", "complexity", "feature_count"}


@dataclass(slots=True)
class Policy:
    name: str
    primary: str = "auroc"
    source: str = "validation"  # "validation" or "cv"
    tolerance: float | str = 0.0  # absolute margin, or "one_se"
    constraints: list[dict[str, Any]] = field(default_factory=list)  # [{"metric": ..., "min": ..., "max": ...}]
    tie_breakers: list[str] = field(default_factory=lambda: ["macro_f1", "brier_score", "mcc"])

    def __post_init__(self) -> None:
        if self.source not in {"validation", "cv"}:
            raise ValueError(f"Unknown policy source: {self.source!r}")
        if isinstance(self.tolerance, str) and self.tolerance != "one_se":
            raise ValueError(f"Unknown tolerance: {self.tolerance!r}")


# the rule used by the first version of the framework: validation AUROC, exact ties broken by Macro-F1, Brier, MCC
LEGACY_AUROC = Policy(name="legacy_auroc")


def load_policy(path: str | Path) -> Policy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Policy(**payload)


def _value(candidate: dict[str, Any], metric: str) -> float:
    if metric == "complexity":
        return float(COMPLEXITY.get(candidate.get("model_family", ""), 99))
    if metric == "feature_count":
        return float(candidate.get("feature_count", 0))
    return float(candidate["metrics"][metric])


def _better_key(candidate: dict[str, Any], metric: str) -> float:
    value = _value(candidate, metric)
    return value if metric in LOWER_IS_BETTER else -value


def _meets(candidate: dict[str, Any], constraint: dict[str, Any]) -> bool:
    value = _value(candidate, constraint["metric"])
    return constraint.get("min", float("-inf")) <= value <= constraint.get("max", float("inf"))


def choose(candidates: list[dict[str, Any]], policy: Policy) -> int:
    """Index of the candidate the policy promotes.

    Each candidate is {"model_family", "feature_count", "is_baseline", "metrics": {...}, "se": {...}} with metrics
    taken from the policy's source. If no candidate meets the constraints, the baseline is kept.
    """
    if not candidates:
        raise ValueError("No candidates to choose from")
    pool = [i for i, c in enumerate(candidates) if all(_meets(c, k) for k in policy.constraints)]
    if not pool:
        baselines = [i for i, c in enumerate(candidates) if c.get("is_baseline")]
        return baselines[0] if baselines else 0

    best = min(pool, key=lambda i: _better_key(candidates[i], policy.primary))
    best_value = _value(candidates[best], policy.primary)
    if policy.tolerance == "one_se":
        margin = float(candidates[best].get("se", {}).get(policy.primary, 0.0))
    else:
        margin = float(policy.tolerance)
    sign = 1.0 if policy.primary in LOWER_IS_BETTER else -1.0
    # eligible: within the declared margin of the best primary value
    eligible = [i for i in pool if sign * (_value(candidates[i], policy.primary) - best_value) <= margin + 1e-12]
    # tie-breakers in declared order, then the primary metric, then the earliest run
    return min(eligible, key=lambda i: tuple(_better_key(candidates[i], m) for m in policy.tie_breakers)
               + (_better_key(candidates[i], policy.primary), i))
