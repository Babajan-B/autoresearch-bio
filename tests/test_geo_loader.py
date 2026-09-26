import gzip
import io
import tarfile

from src.data_adapters.geo_loader import GEODataAdapter


class MockResponse:
    def __init__(self, payload: bytes) -> None:
        self.content = payload

    def raise_for_status(self) -> None:
        return None


class MockSession:
    def __init__(self, mapping: dict[str, bytes]) -> None:
        self.mapping = mapping

    def get(self, url: str, timeout: int = 60) -> MockResponse:
        return MockResponse(self.mapping[url])


def _build_miniml_tgz() -> bytes:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<MINiML>
  <Series>
    <Title>Mock GEO study</Title>
    <Summary>Mock summary</Summary>
  </Series>
  <Sample>
    <Accession>GSM1</Accession>
    <Title>Sample 1</Title>
    <Source>tissue</Source>
    <Characteristics>condition: case</Characteristics>
  </Sample>
  <Sample>
    <Accession>GSM2</Accession>
    <Title>Sample 2</Title>
    <Source>tissue</Source>
    <Characteristics>condition: control</Characteristics>
  </Sample>
</MINiML>
"""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("GSE12345_family.xml")
        payload = xml.encode("utf-8")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _build_series_matrix() -> bytes:
    text = """!Sample_geo_accession\t\"GSM1\"\t\"GSM2\"
!Sample_title\t\"Sample 1\"\t\"Sample 2\"
!Sample_characteristics_ch1\t\"condition: case\"\t\"condition: control\"
!series_matrix_table_begin
ID_REF\tGSM1\tGSM2
ESR1\t5.0\t1.0
PGR\t4.0\t1.2
GENE_X\t2.0\t3.0
!series_matrix_table_end
"""
    return gzip.compress(text.encode("utf-8"))


def test_geo_loader_returns_standard_bundle() -> None:
    accession = "GSE12345"
    base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12345"
    session = MockSession(
        {
            f"{base}/miniml/GSE12345_family.xml.tgz": _build_miniml_tgz(),
            f"{base}/matrix/GSE12345_series_matrix.txt.gz": _build_series_matrix(),
        }
    )

    bundle = GEODataAdapter(session=session).load(accession=accession, label_column="condition")
    assert bundle.metadata["accession"] == accession
    assert bundle.expression.shape == (2, 3)
    assert bundle.phenotype.loc["GSM1", "condition"] == "case"


def test_geo_loader_can_align_expression_by_sample_title() -> None:
    accession = "GSE12345"
    base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12345"
    session = MockSession(
        {
            f"{base}/miniml/GSE12345_family.xml.tgz": _build_miniml_tgz(),
            f"{base}/matrix/GSE12345_series_matrix.txt.gz": gzip.compress(
                """!Sample_geo_accession\t\"GSM1\"\t\"GSM2\"
!Sample_title\t\"Sample 1\"\t\"Sample 2\"
!series_matrix_table_begin
ID_REF\tSample 1\tSample 2
ESR1\t5.0\t1.0
GENE_X\t2.0\t3.0
!series_matrix_table_end
""".encode("utf-8")
            ),
        }
    )

    bundle = GEODataAdapter(session=session).load(accession=accession, label_column="condition")
    assert bundle.expression.index.tolist() == ["Sample 1", "Sample 2"]
    assert bundle.phenotype.loc["Sample 1", "geo_accession"] == "GSM1"


def test_geo_loader_falls_back_when_series_matrix_has_zero_genes() -> None:
    accession = "GSE12345"
    base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12345"
    session = MockSession(
        {
            f"{base}/miniml/GSE12345_family.xml.tgz": _build_miniml_tgz(),
            f"{base}/matrix/GSE12345_series_matrix.txt.gz": gzip.compress(
                """!Sample_geo_accession\t\"GSM1\"\t\"GSM2\"
!Sample_title\t\"Sample 1\"\t\"Sample 2\"
!series_matrix_table_begin
ID_REF\tGSM1\tGSM2
!series_matrix_table_end
""".encode("utf-8")
            ),
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12345/suppl/mock_expression.csv.gz": gzip.compress(
                "gene,GSM1,GSM2\nESR1,5.0,1.0\nGENE_X,2.0,3.0\n".encode("utf-8")
            ),
        }
    )

    bundle = GEODataAdapter(session=session).load(
        accession=accession,
        label_column="condition",
        supplementary_urls=["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12345/suppl/mock_expression.csv.gz"],
    )
    assert bundle.expression.shape == (2, 2)
    assert bundle.expression.columns.tolist() == ["ESR1", "GENE_X"]
