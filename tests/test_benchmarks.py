from src.benchmarks import resolve_benchmark_preset


def test_benchmark_preset_gse96058_er() -> None:
    preset = resolve_benchmark_preset("gse96058_er")
    assert preset["accession"] == "GSE96058"
    assert preset["label_column"] == "er status"
    assert preset["exclude_title_suffixes"] == ["repl"]
