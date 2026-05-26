from pathlib import Path

from seqlens import SeqLens


def test_diagnose_sample_stock() -> None:
    root = Path(__file__).resolve().parents[1]
    project = SeqLens(root / "examples" / "sample_stock.csv", time_col="date", target_col="close")

    report = project.diagnose()

    assert report.row_count == 12
    assert report.valid_timestamp_ratio == 1
