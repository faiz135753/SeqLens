from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    """Defines reusable factor engineering rules for tabular and sequence models."""

    lag_columns: list[str] = field(default_factory=list)
    lag_periods: list[int] = field(default_factory=list)
    rolling_columns: list[str] = field(default_factory=list)
    rolling_windows: list[int] = field(default_factory=list)
    rolling_stats: list[str] = field(default_factory=lambda: ["mean"])
    diff_columns: list[str] = field(default_factory=list)
    diff_periods: list[int] = field(default_factory=list)
    ratio_to_rolling_mean_columns: list[str] = field(default_factory=list)
    ratio_windows: list[int] = field(default_factory=list)
    calendar: list[str] = field(default_factory=list)


def build_factors(frame: pd.DataFrame, *, time_col: str, spec: FactorSpec) -> pd.DataFrame:
    factors = pd.DataFrame(index=frame.index)

    for column in spec.lag_columns:
        series = _numeric_column(frame, column)
        for period in spec.lag_periods:
            _validate_positive(period, "lag period")
            factors[f"{column}_lag_{period}"] = series.shift(period)

    for column in spec.rolling_columns:
        series = _numeric_column(frame, column)
        for window in spec.rolling_windows:
            _validate_positive(window, "rolling window")
            rolling = series.shift(1).rolling(window=window, min_periods=window)
            for stat in spec.rolling_stats:
                factors[f"{column}_roll_{window}_{stat}"] = _rolling_stat(rolling, stat)

    for column in spec.diff_columns:
        series = _numeric_column(frame, column)
        for period in spec.diff_periods:
            _validate_positive(period, "difference period")
            factors[f"{column}_diff_{period}"] = series - series.shift(period)

    for column in spec.ratio_to_rolling_mean_columns:
        series = _numeric_column(frame, column)
        for window in spec.ratio_windows:
            _validate_positive(window, "ratio window")
            denominator = series.shift(1).rolling(window=window, min_periods=window).mean()
            factors[f"{column}_ratio_roll_{window}_mean"] = series / denominator

    if spec.calendar:
        time = pd.to_datetime(frame[time_col], errors="coerce")
        for field_name in spec.calendar:
            factors[field_name] = _calendar_field(time, field_name)

    return factors


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Factor column not found: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def _rolling_stat(rolling: pd.core.window.Rolling, stat: str) -> pd.Series:
    if stat == "mean":
        return rolling.mean()
    if stat == "sum":
        return rolling.sum()
    if stat == "max":
        return rolling.max()
    if stat == "min":
        return rolling.min()
    if stat == "std":
        return rolling.std()
    if stat == "nonzero_count":
        return rolling.apply(lambda values: (values != 0).sum(), raw=True)
    if stat == "slope":
        return rolling.apply(_window_slope, raw=True)
    raise ValueError(f"Unsupported rolling stat: {stat}")


def _window_slope(values) -> float:
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / (len(values) - 1)


def _calendar_field(time: pd.Series, field_name: str) -> pd.Series:
    if field_name == "hour":
        return time.dt.hour
    if field_name == "day_of_week":
        return time.dt.dayofweek
    if field_name == "month":
        return time.dt.month
    if field_name == "quarter":
        return time.dt.quarter
    if field_name == "season":
        month = time.dt.month
        return ((month % 12) // 3) + 1
    raise ValueError(f"Unsupported calendar field: {field_name}")


def _validate_positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
