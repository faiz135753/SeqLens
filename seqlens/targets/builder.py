from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    """Defines how raw observations become a supervised prediction target."""

    type: str
    column: str
    horizon: int = 1
    threshold: float | None = None
    aggregation: str = "value"
    name: str | None = None

    @property
    def output_name(self) -> str:
        if self.name:
            return self.name
        if self.type == "future_value":
            return f"{self.column}_t_plus_{self.horizon}"
        if self.type == "future_return":
            return f"{self.column}_return_t_plus_{self.horizon}"
        if self.type == "future_direction":
            return f"{self.column}_direction_t_plus_{self.horizon}"
        if self.type == "future_window_event":
            return f"{self.column}_{self.aggregation}_next_{self.horizon}_event"
        return f"{self.column}_{self.type}"


def build_target(frame: pd.DataFrame, spec: TargetSpec) -> pd.Series:
    if spec.column not in frame.columns:
        raise ValueError(f"Target column not found: {spec.column}")
    if spec.horizon < 1:
        raise ValueError("Target horizon must be at least 1.")

    values = pd.to_numeric(frame[spec.column], errors="coerce")

    if spec.type == "future_value":
        return values.shift(-spec.horizon).rename(spec.output_name)
    if spec.type == "future_return":
        future = values.shift(-spec.horizon)
        return ((future / values) - 1).rename(spec.output_name)
    if spec.type == "future_direction":
        future = values.shift(-spec.horizon)
        target = (future > values).astype("Int64")
        target[future.isna() | values.isna()] = pd.NA
        return target.rename(spec.output_name)
    if spec.type == "future_window_event":
        if spec.threshold is None:
            raise ValueError("future_window_event requires a threshold.")
        future_window = _future_window(values, spec.horizon, spec.aggregation)
        target = (future_window >= spec.threshold).astype("Int64")
        target[future_window.isna()] = pd.NA
        return target.rename(spec.output_name)

    raise ValueError(f"Unsupported target type: {spec.type}")


def _future_window(values: pd.Series, horizon: int, aggregation: str) -> pd.Series:
    shifted = [values.shift(-step) for step in range(1, horizon + 1)]
    future = pd.concat(shifted, axis=1)
    if aggregation == "sum":
        return future.sum(axis=1, min_count=horizon)
    if aggregation == "max":
        return future.max(axis=1, skipna=False)
    if aggregation == "mean":
        return future.mean(axis=1, skipna=False)
    if aggregation == "value":
        return values.shift(-horizon)
    raise ValueError(f"Unsupported future window aggregation: {aggregation}")
