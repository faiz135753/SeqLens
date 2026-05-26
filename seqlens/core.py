from __future__ import annotations

from pathlib import Path

from seqlens.data.loader import TimeSeriesDataset, load_csv
from seqlens.diagnostics.timeseries import DiagnosticReport, diagnose_timeseries


class SeqLens:
    """High-level entry point for a SeqLens automated experiment workflow."""

    def __init__(self, data: str | Path, time_col: str, target_col: str):
        self.data_path = Path(data)
        self.time_col = time_col
        self.target_col = target_col
        self.dataset: TimeSeriesDataset | None = None
        self.diagnostic_report: DiagnosticReport | None = None

    def load(self) -> TimeSeriesDataset:
        self.dataset = load_csv(self.data_path, self.time_col, self.target_col)
        return self.dataset

    def diagnose(self) -> DiagnosticReport:
        dataset = self.dataset or self.load()
        self.diagnostic_report = diagnose_timeseries(dataset)
        return self.diagnostic_report
