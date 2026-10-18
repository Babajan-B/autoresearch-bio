from __future__ import annotations

import gzip
import io
import json
import subprocess
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TABULAR_SUFFIXES = {".tsv", ".txt", ".csv", ".gct", ".xlsx", ".xls"}


@dataclass(slots=True)
class DatasetBundle:
    expression: pd.DataFrame
    phenotype: pd.DataFrame
    metadata: dict[str, Any]
    label_column: str

    @property
    def dataset_id(self) -> str:
        return str(self.metadata.get("dataset_id", "dataset"))


@dataclass(slots=True)
class PreparedDataset:
    bundle: DatasetBundle
    splits: dict[str, list[str]]
    split_seed: int
    normalization: str


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_delimiter(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".csv"):
        return ","
    return "\t"


def _decode_bytes(payload: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="ignore")


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(table_path)
    delimiter = infer_delimiter(table_path.name)
    return pd.read_csv(table_path, sep=delimiter, engine="python")


def read_table_from_bytes(payload: bytes, name_hint: str) -> pd.DataFrame:
    lowered = name_hint.lower()

    if lowered.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if lowered.endswith((".tar.gz", ".tgz")) else "r"
        with tarfile.open(fileobj=io.BytesIO(payload), mode=mode) as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and any(member.name.lower().endswith(ext) for ext in TABULAR_SUFFIXES)
            ]
            if not members:
                raise ValueError(f"No tabular file found inside archive {name_hint}")
            extracted = archive.extractfile(members[0])
            if extracted is None:
                raise ValueError(f"Failed to extract {members[0].name} from {name_hint}")
            return read_table_from_bytes(extracted.read(), members[0].name)

    if lowered.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                name
                for name in archive.namelist()
                if any(name.lower().endswith(ext) for ext in TABULAR_SUFFIXES)
            ]
            if not members:
                raise ValueError(f"No tabular file found inside archive {name_hint}")
            with archive.open(members[0]) as extracted:
                return read_table_from_bytes(extracted.read(), members[0])

    if lowered.endswith(".gz"):
        return read_table_from_bytes(gzip.decompress(payload), lowered[:-3])

    if lowered.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(payload))

    delimiter = infer_delimiter(name_hint)
    text = _decode_bytes(payload)
    return pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str | Path, default: Any | None = None) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        return default
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "dataset"


def save_prepared_dataset(prepared: PreparedDataset, output_dir: str | Path) -> Path:
    output_path = ensure_dir(output_dir)
    prepared.bundle.expression.to_csv(output_path / "expression.csv.gz", index=True, compression="gzip")
    prepared.bundle.phenotype.to_csv(output_path / "phenotype.csv.gz", index=True, compression="gzip")
    write_json(output_path / "metadata.json", prepared.bundle.metadata)
    write_json(output_path / "splits.json", prepared.splits)
    write_json(
        output_path / "manifest.json",
        {
            "dataset_id": prepared.bundle.dataset_id,
            "label_column": prepared.bundle.label_column,
            "normalization": prepared.normalization,
            "split_seed": prepared.split_seed,
            "created_at": utc_now(),
        },
    )
    return output_path


def load_prepared_dataset(dataset_dir: str | Path) -> PreparedDataset:
    dataset_path = Path(dataset_dir)
    expression = pd.read_csv(dataset_path / "expression.csv.gz", index_col=0)
    phenotype = pd.read_csv(dataset_path / "phenotype.csv.gz", index_col=0)
    metadata = read_json(dataset_path / "metadata.json", default={})
    splits = read_json(dataset_path / "splits.json", default={})
    manifest = read_json(dataset_path / "manifest.json", default={})
    label_column = str(manifest.get("label_column", "label"))
    return PreparedDataset(
        bundle=DatasetBundle(expression=expression, phenotype=phenotype, metadata=metadata, label_column=label_column),
        splits=splits,
        split_seed=int(manifest.get("split_seed", 42)),
        normalization=str(manifest.get("normalization", "auto")),
    )


def get_git_commit_hash(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def serialize_dataclass(instance: Any) -> dict[str, Any]:
    return asdict(instance)
