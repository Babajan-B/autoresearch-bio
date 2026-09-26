from __future__ import annotations

import csv
import gzip
import io
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils.io import DatasetBundle, read_table_from_bytes
from src.utils.validation import orient_expression_table, validate_geo_accession


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    value = element.text.strip()
    return value or None


def _series_bucket(accession: str) -> str:
    return f"GSE{accession[3:-3]}nnn"


def _rank_supplementary_urls(urls: list[str]) -> list[str]:
    normalized_urls = [_normalize_supplementary_url(url) for url in urls]

    def sort_key(url: str) -> tuple[int, int]:
        lowered = url.lower()
        score = 0
        for token in ("matrix", "count", "expr", "expression", "fpkm", "tpm", "rsem"):
            if token in lowered:
                score += 1
        return (-score, len(url))

    return sorted(dict.fromkeys(normalized_urls), key=sort_key)


def _normalize_supplementary_url(url: str) -> str:
    normalized = url.strip()
    if normalized.lower().startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return "https://" + normalized[len("ftp://") :]
    return normalized


@dataclass(slots=True)
class _GeoMetadata:
    phenotype: pd.DataFrame
    series_metadata: dict[str, Any]
    supplementary_urls: list[str]


class GEODataAdapter:
    def __init__(
        self,
        session: requests.Session | None = None,
        base_url: str = "https://ftp.ncbi.nlm.nih.gov/geo",
        timeout_seconds: int = 60,
    ) -> None:
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def load(
        self,
        accession: str,
        label_column: str,
        supplementary_urls: list[str] | None = None,
    ) -> DatasetBundle:
        geo_accession = validate_geo_accession(accession)
        print(f"[geo] loading metadata for {geo_accession}", flush=True)
        metadata = self._load_miniml_metadata(geo_accession)

        expression = None
        try:
            print(f"[geo] trying series matrix for {geo_accession}", flush=True)
            expression, matrix_phenotype = self._load_series_matrix(geo_accession)
            if expression.shape[1] == 0:
                raise ValueError(f"GEO series matrix for {geo_accession} did not contain expression features")
            phenotype = metadata.phenotype.combine_first(matrix_phenotype)
        except Exception:
            print(f"[geo] series matrix unavailable or unusable for {geo_accession}", flush=True)
            expression = None
            phenotype = metadata.phenotype

        if expression is None:
            candidate_supplementary_urls = supplementary_urls or metadata.supplementary_urls or self._load_series_page_supplementary_urls(geo_accession)
            print(f"[geo] trying {len(candidate_supplementary_urls)} supplementary candidate files for {geo_accession}", flush=True)
            expression = self._load_from_supplementary(candidate_supplementary_urls, phenotype)

        if expression is None:
            raise ValueError(f"Could not resolve processed expression data for {geo_accession}")

        expression, phenotype = self._align_expression_and_phenotype(expression, phenotype)
        print(f"[geo] aligned expression and phenotype for {geo_accession}: {expression.shape[0]} samples x {expression.shape[1]} genes", flush=True)

        dataset_metadata = dict(metadata.series_metadata)
        dataset_metadata.update(
            {
                "dataset_id": geo_accession,
                "source": "geo",
                "accession": geo_accession,
            }
        )
        return DatasetBundle(expression=expression, phenotype=phenotype, metadata=dataset_metadata, label_column=label_column)

    def _download(self, url: str) -> bytes:
        print(f"[geo] downloading {url}", flush=True)
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        print(f"[geo] downloaded {len(response.content)} bytes from {url}", flush=True)
        return response.content

    def _load_miniml_metadata(self, accession: str) -> _GeoMetadata:
        bucket = _series_bucket(accession)
        url = f"{self.base_url}/series/{bucket}/{accession}/miniml/{accession}_family.xml.tgz"
        payload = self._download(url)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            member = next((member for member in archive.getmembers() if member.name.endswith(".xml")), None)
            if member is None:
                raise ValueError(f"MINiML archive for {accession} did not contain XML metadata")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Failed to extract MINiML XML for {accession}")
            root = ET.fromstring(extracted.read())

        series_title = None
        series_summary = None
        series_supplementary: list[str] = []
        sample_rows: list[dict[str, Any]] = []
        sample_supplementary: list[str] = []

        for child in root:
            name = _tag_name(child)
            if name == "Series":
                series_title = _text(next((item for item in child if _tag_name(item) == "Title"), None))
                series_summary = _text(next((item for item in child if _tag_name(item) == "Summary"), None))
                for item in child.iter():
                    if _tag_name(item) == "Supplementary-Data" and _text(item):
                        series_supplementary.append(_text(item) or "")
            elif name == "Sample":
                accession_value = _text(next((item for item in child if _tag_name(item) == "Accession"), None))
                title_value = _text(next((item for item in child if _tag_name(item) == "Title"), None))
                source_value = _text(next((item for item in child if _tag_name(item) == "Source"), None))
                row: dict[str, Any] = {
                    "geo_accession": accession_value or child.attrib.get("iid") or "",
                    "title": title_value,
                    "source_name": source_value,
                }
                characteristic_counter = 0
                for item in child.iter():
                    tag = _tag_name(item)
                    if tag == "Characteristics":
                        characteristic_counter += 1
                        text_value = _text(item)
                        if not text_value:
                            continue
                        characteristic_tag = item.attrib.get("tag")
                        if characteristic_tag:
                            row[characteristic_tag.strip().lower()] = text_value
                            continue
                        if ":" in text_value:
                            key, value = text_value.split(":", 1)
                            row[key.strip().lower()] = value.strip()
                        else:
                            row[f"characteristics_{characteristic_counter}"] = text_value
                    elif tag == "Supplementary-Data" and _text(item):
                        sample_supplementary.append(_text(item) or "")
                sample_rows.append(row)

        phenotype = pd.DataFrame(sample_rows).set_index("geo_accession", drop=False)
        phenotype.index = phenotype.index.astype(str)
        return _GeoMetadata(
            phenotype=phenotype,
            series_metadata={
                "title": series_title,
                "summary": series_summary,
                "supplementary_urls": sorted(dict.fromkeys(series_supplementary + sample_supplementary)),
            },
            supplementary_urls=sorted(dict.fromkeys(series_supplementary + sample_supplementary)),
        )

    def _load_series_matrix(self, accession: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        bucket = _series_bucket(accession)
        url = f"{self.base_url}/series/{bucket}/{accession}/matrix/{accession}_series_matrix.txt.gz"
        payload = self._download(url)
        text = gzip.decompress(payload).decode("utf-8", errors="ignore")
        lines = [line for line in text.splitlines() if line]

        sample_meta: dict[str, list[str]] = {}
        table_lines: list[str] = []
        inside_table = False
        for line in lines:
            if line.startswith("!sample_table_begin") or line.startswith("!series_matrix_table_begin"):
                inside_table = True
                continue
            if line.startswith("!sample_table_end") or line.startswith("!series_matrix_table_end"):
                inside_table = False
                continue
            if inside_table:
                table_lines.append(line)
                continue
            if line.startswith("!Sample_"):
                key, *values = next(csv.reader([line], delimiter="\t"))
                sample_meta[key.replace("!Sample_", "")] = [value.strip('"') for value in values]

        if not table_lines:
            raise ValueError(f"GEO series matrix for {accession} did not contain an expression table")

        reader = csv.reader(table_lines, delimiter="\t")
        rows = list(reader)
        matrix = pd.DataFrame(rows[1:], columns=rows[0])
        matrix = matrix.set_index(matrix.columns[0])
        expression = matrix.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").transpose()
        expression.index = expression.index.astype(str)

        phenotype = pd.DataFrame(sample_meta)
        if "geo_accession" in phenotype.columns:
            phenotype.index = phenotype["geo_accession"].astype(str)
        elif "title" in phenotype.columns:
            phenotype.index = phenotype["title"].astype(str)
        else:
            phenotype.index = expression.index

        for column in list(phenotype.columns):
            if column.startswith("characteristics"):
                expanded = phenotype[column].astype(str).str.split(":", n=1, expand=True)
                if expanded.shape[1] == 2:
                    phenotype[expanded[0].str.strip().iloc[0]] = expanded[1].str.strip()

        return expression, phenotype

    def _load_series_page_supplementary_urls(self, accession: str) -> list[str]:
        page_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
        response = self.session.get(page_url, timeout=self.timeout_seconds)
        response.raise_for_status()
        html = response.text
        bucket = _series_bucket(accession)
        filename_pattern = re.compile(rf"{re.escape(accession)}[A-Za-z0-9_.\-]*\.(?:csv|txt|tsv|gct)(?:\.gz)?", re.IGNORECASE)
        filenames = sorted(set(filename_pattern.findall(html)))
        return [
            f"{self.base_url}/series/{bucket}/{accession}/suppl/{filename}"
            for filename in filenames
        ]

    def _load_from_supplementary(self, urls: list[str], phenotype: pd.DataFrame) -> pd.DataFrame | None:
        for url in _rank_supplementary_urls(urls):
            try:
                print(f"[geo] probing supplementary file {url}", flush=True)
                payload = self._download(url)
                table = read_table_from_bytes(payload, Path(url).name)
                for sample_ids in self._sample_id_candidates(phenotype):
                    try:
                        print(f"[geo] attempting sample alignment for {Path(url).name} using {len(sample_ids)} candidate IDs", flush=True)
                        return orient_expression_table(table, sample_ids)
                    except Exception:
                        continue
            except Exception:
                print(f"[geo] failed to use supplementary file {url}", flush=True)
                continue
        return None

    def _sample_id_candidates(self, phenotype: pd.DataFrame) -> list[list[str]]:
        candidates: list[list[str]] = [phenotype.index.astype(str).tolist()]
        for column in ("title", "geo_accession"):
            if column in phenotype.columns:
                series = phenotype[column].astype(str).str.strip()
                if series.nunique() == len(series):
                    candidates.append(series.tolist())
        return candidates

    def _align_expression_and_phenotype(
        self,
        expression: pd.DataFrame,
        phenotype: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        candidate_columns = ["title", "geo_accession"]
        for column in candidate_columns:
            if column not in phenotype.columns:
                continue
            ids = phenotype[column].astype(str).str.strip()
            if ids.nunique() != len(ids):
                continue
            overlap = len(set(expression.index).intersection(set(ids)))
            if overlap == 0:
                continue
            reindexed = phenotype.assign(_sample_key=ids).set_index("_sample_key")
            shared = expression.index.intersection(reindexed.index)
            return expression.loc[shared], reindexed.loc[shared]

        shared = expression.index.intersection(phenotype.index)
        return expression.loc[shared], phenotype.loc[shared]


def load_geo_dataset(
    accession: str,
    label_column: str,
    supplementary_urls: list[str] | None = None,
    session: requests.Session | None = None,
) -> DatasetBundle:
    return GEODataAdapter(session=session).load(
        accession=accession,
        label_column=label_column,
        supplementary_urls=supplementary_urls,
    )
