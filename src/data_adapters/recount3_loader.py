from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils.io import DatasetBundle, read_table_from_bytes
from src.utils.validation import orient_expression_table, validate_recount3_project


def _project_bucket(project: str) -> str:
    return project[-2:]


def _candidate_sample_id_columns(frame: pd.DataFrame) -> list[str]:
    candidates = ["external_id", "run", "sample", "rail_id", "sra.run", "sample_id"]
    return [column for column in candidates if column in frame.columns]


class Recount3DataAdapter:
    def __init__(
        self,
        session: requests.Session | None = None,
        base_url: str = "http://duffel.rail.bio/recount3",
        timeout_seconds: int = 60,
    ) -> None:
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def load(
        self,
        project: str,
        label_column: str,
        organism: str = "human",
        project_home: str = "data_sources/sra",
        annotation: str = "G026",
    ) -> DatasetBundle:
        recount_project = validate_recount3_project(project)
        source_name = Path(project_home).name
        bucket = _project_bucket(recount_project)

        metadata_url = (
            f"{self.base_url}/{organism}/{project_home}/metadata/{bucket}/{recount_project}/"
            f"{source_name}.sra.{recount_project}.MD.gz"
        )
        gene_url = (
            f"{self.base_url}/{organism}/{project_home}/gene_sums/{bucket}/{recount_project}/"
            f"{source_name}.gene_sums.{recount_project}.{annotation}.gz"
        )

        metadata = read_table_from_bytes(self._download(metadata_url), Path(metadata_url).name)
        counts = read_table_from_bytes(self._download(gene_url), Path(gene_url).name)

        phenotype = self._normalize_metadata(metadata)
        expression = orient_expression_table(counts, phenotype.index)

        dataset_metadata: dict[str, Any] = {
            "dataset_id": recount_project,
            "source": "recount3",
            "project": recount_project,
            "organism": organism,
            "project_home": project_home,
            "annotation": annotation,
            "metadata_url": metadata_url,
            "gene_sums_url": gene_url,
        }
        return DatasetBundle(expression=expression, phenotype=phenotype, metadata=dataset_metadata, label_column=label_column)

    def _download(self, url: str) -> bytes:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.content

    def _normalize_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        frame = metadata.copy()
        for candidate in _candidate_sample_id_columns(frame):
            candidate_values = frame[candidate].astype(str).str.strip()
            if candidate_values.nunique() == len(frame):
                frame = frame.assign(sample_id=candidate_values).set_index("sample_id")
                return frame
        raise ValueError("Could not identify a unique sample ID column in recount3 metadata")


def load_recount3_dataset(
    project: str,
    label_column: str,
    organism: str = "human",
    project_home: str = "data_sources/sra",
    annotation: str = "G026",
    session: requests.Session | None = None,
) -> DatasetBundle:
    return Recount3DataAdapter(session=session).load(
        project=project,
        label_column=label_column,
        organism=organism,
        project_home=project_home,
        annotation=annotation,
    )
