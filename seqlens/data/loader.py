from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesDataset:
    frame: pd.DataFrame
    time_col: str
    target_col: str

    @property
    def target(self) -> pd.Series:
        return self.frame[self.target_col]

    @property
    def timestamps(self) -> pd.Series:
        return self.frame[self.time_col]


def load_csv(path: str | Path, time_col: str, target_col: str) -> TimeSeriesDataset:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    _validate_columns(frame, time_col, target_col)

    frame = frame.copy()
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.sort_values(time_col).reset_index(drop=True)

    return TimeSeriesDataset(frame=frame, time_col=time_col, target_col=target_col)


def _validate_columns(frame: pd.DataFrame, time_col: str, target_col: str) -> None:
    missing = [col for col in [time_col, target_col] if col not in frame.columns]
    if missing:
        available = ", ".join(frame.columns)
        raise ValueError(f"Missing required columns {missing}. Available columns: {available}")

