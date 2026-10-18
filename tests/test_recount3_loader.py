import gzip

from src.data_adapters.recount3_loader import Recount3DataAdapter


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


def test_recount3_loader_harmonizes_metadata_and_counts() -> None:
    metadata = "external_id\tgroup\nSRR1\tcase\nSRR2\tcontrol\n"
    counts = "gene_id\tSRR1\tSRR2\nESR1\t10\t1\nGENE_X\t4\t5\n"
    project = "SRP009615"
    base = "http://duffel.rail.bio/recount3/human/data_sources/sra"
    bucket = "15"
    session = MockSession(
        {
            f"{base}/metadata/{bucket}/{project}/sra.sra.{project}.MD.gz": gzip.compress(metadata.encode("utf-8")),
            f"{base}/gene_sums/{bucket}/{project}/sra.gene_sums.{project}.G026.gz": gzip.compress(counts.encode("utf-8")),
        }
    )

    bundle = Recount3DataAdapter(session=session).load(project=project, label_column="group")
    assert bundle.expression.shape == (2, 2)
    assert bundle.phenotype.index.tolist() == ["SRR1", "SRR2"]
    assert bundle.phenotype.loc["SRR1", "group"] == "case"
