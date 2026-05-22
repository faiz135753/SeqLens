from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from seqlens.data.loader import TimeSeriesDataset


@dataclass(frozen=True)
class DiagnosticIssue:
    level: str
    message: str


@dataclass(frozen=True)
class DiagnosticReport:
    row_count: int
    valid_timestamp_ratio: float
    target_missing_ratio: float
    duplicate_timestamp_count: int
    inferred_frequency: str | None
    is_regular_interval: bool
    outlier_ratio: float
    lag1_autocorrelation: float | None
    trend_strength: float | None
    issues: list[DiagnosticIssue] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Rows: {self.row_count}",
            f"Valid timestamps: {self.valid_timestamp_ratio:.1%}",
            f"Target missing values: {self.target_missing_ratio:.1%}",
            f"Duplicate timestamps: {self.duplicate_timestamp_count}",
            f"Inferred frequency: {self.inferred_frequency or 'unknown'}",
            f"Regular interval: {self.is_regular_interval}",
            f"Outlier ratio: {self.outlier_ratio:.1%}",
        ]
        if self.lag1_autocorrelation is not None:
            lines.append(f"Lag-1 autocorrelation: {self.lag1_autocorrelation:.3f}")
        if self.trend_strength is not None:
            lines.append(f"Trend strength: {self.trend_strength:.3f}")
        if self.issues:
            lines.append("Issues:")
            lines.extend(f"- [{issue.level}] {issue.message}" for issue in self.issues)
        return "\n".join(lines)


def diagnose_timeseries(dataset: TimeSeriesDataset) -> DiagnosticReport:
    frame = dataset.frame
    time = frame[dataset.time_col]
    target = frame[dataset.target_col]

    row_count = len(frame)
    valid_timestamp_ratio = float(time.notna().mean()) if row_count else 0.0
    target_missing_ratio = float(target.isna().mean()) if row_count else 0.0
    duplicate_timestamp_count = int(time.duplicated().sum())

    valid_time = time.dropna()
    inferred_frequency = _infer_frequency(valid_time)
    is_regular_interval = _is_regular_interval(valid_time)

    clean_target = target.dropna()
    outlier_ratio = _outlier_ratio(clean_target)
    lag1_autocorrelation = _lag1_autocorrelation(clean_target)
    trend_strength = _trend_strength(clean_target)

    issues = _build_issues(
        row_count=row_count,
        valid_timestamp_ratio=valid_timestamp_ratio,
        target_missing_ratio=target_missing_ratio,
        duplicate_timestamp_count=duplicate_timestamp_count,
        is_regular_interval=is_regular_interval,
        outlier_ratio=outlier_ratio,
        lag1_autocorrelation=lag1_autocorrelation,
    )

    return DiagnosticReport(
        row_count=row_count,
        valid_timestamp_ratio=valid_timestamp_ratio,
        target_missing_ratio=target_missing_ratio,
        duplicate_timestamp_count=duplicate_timestamp_count,
        inferred_frequency=inferred_frequency,
        is_regular_interval=is_regular_interval,
        outlier_ratio=outlier_ratio,
        lag1_autocorrelation=lag1_autocorrelation,
        trend_strength=trend_strength,
        issues=issues,
    )


def _infer_frequency(time: pd.Series) -> str | None:
    if len(time) < 3:
        return None
    return pd.infer_freq(time)


def _is_regular_interval(time: pd.Series) -> bool:
    if len(time) < 3:
        return False
    deltas = time.sort_values().diff().dropna()
    return bool(len(deltas) > 0 and deltas.nunique() == 1)


def _outlier_ratio(series: pd.Series) -> float:
    if len(series) < 4:
        return 0.0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((series < lower) | (series > upper)).mean())


def _lag1_autocorrelation(series: pd.Series) -> float | None:
    if len(series) < 3 or series.nunique() <= 1:
        return None
    value = series.autocorr(lag=1)
    if pd.isna(value):
        return None
    return float(value)


def _trend_strength(series: pd.Series) -> float | None:
    if len(series) < 3 or series.nunique() <= 1:
        return None
    x = np.arange(len(series), dtype=float)
    y = series.to_numpy(dtype=float)
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return None
    return float(abs(corr))


def _build_issues(
    *,
    row_count: int,
    valid_timestamp_ratio: float,
    target_missing_ratio: float,
    duplicate_timestamp_count: int,
    is_regular_interval: bool,
    outlier_ratio: float,
    lag1_autocorrelation: float | None,
) -> list[DiagnosticIssue]:
    issues: list[DiagnosticIssue] = []
    if row_count < 200:
        issues.append(
            DiagnosticIssue("warning", "The series is short for LSTM training; compare simple baselines first.")
        )
    if valid_timestamp_ratio < 1:
        issues.append(DiagnosticIssue("error", "Some timestamps could not be parsed."))
    if target_missing_ratio > 0:
        issues.append(DiagnosticIssue("warning", "The target column contains missing or non-numeric values."))
    if duplicate_timestamp_count:
        issues.append(DiagnosticIssue("warning", "Duplicate timestamps may create leakage or ordering issues."))
    if not is_regular_interval:
        issues.append(DiagnosticIssue("warning", "The time interval is irregular or too short to infer reliably."))
    if outlier_ratio > 0.05:
        issues.append(DiagnosticIssue("info", "Outlier ratio is above 5%; inspect shocks or data quality."))
    if lag1_autocorrelation is not None and abs(lag1_autocorrelation) < 0.2:
        issues.append(DiagnosticIssue("info", "Weak lag-1 autocorrelation may limit LSTM usefulness."))
    return issues

