from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from src.utils.io import get_git_commit_hash, git_tree_is_dirty, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "shap")


def collect_provenance(prepared_dir: str | Path, declaration_path: str | Path | None = None,
                       require_clean: bool = False) -> dict[str, Any]:
    """Everything needed to tie a run to the exact code, data split and declared plan that produced it."""
    commit, dirty = get_git_commit_hash(REPO_ROOT), git_tree_is_dirty(REPO_ROOT)
    if require_clean and (commit is None or dirty):
        raise RuntimeError("Refusing to run: code is not in a clean git working tree (commit your changes first)")
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    splits_path = Path(prepared_dir) / "splits.json"
    record: dict[str, Any] = {
        "git_commit_hash": commit,
        "git_tree_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "splits_path": str(splits_path),
        "splits_sha256": sha256_file(splits_path) if splits_path.exists() else None,
    }
    if declaration_path is not None:
        if not Path(declaration_path).exists():
            raise FileNotFoundError(f"Declaration file not found: {declaration_path}")
        record["declaration_path"] = str(declaration_path)
        record["declaration_sha256"] = sha256_file(declaration_path)
    return record
