from __future__ import annotations

from dataclasses import dataclass

from seqlens.factors import FactorSpec
from seqlens.targets import TargetSpec
from seqlens.windows import WindowSpec


@dataclass(frozen=True)
class DomainPreset:
    name: str
    task_type: str
    target: TargetSpec
    factors: FactorSpec
    windows: list[WindowSpec]
    primary_metric: str


def rainfall_extreme_rain_preset(
    *,
    rain_col: str = "rainfall",
    horizon: int = 3,
    threshold: float = 80,
) -> DomainPreset:
    return DomainPreset(
        name="rainfall_extreme_rain",
        task_type="classification",
        target=TargetSpec(
            type="future_window_event",
            column=rain_col,
            horizon=horizon,
            aggregation="sum",
            threshold=threshold,
            name=f"heavy_rain_next_{horizon}h",
        ),
        factors=FactorSpec(
            lag_columns=[rain_col],
            lag_periods=[1, 2, 3, 6, 12, 24],
            rolling_columns=[rain_col],
            rolling_windows=[3, 6, 12, 24],
            rolling_stats=["sum", "max", "mean", "std"],
            calendar=["hour", "month", "season"],
        ),
        windows=[
            WindowSpec(observation=12, horizon=horizon),
            WindowSpec(observation=24, horizon=horizon),
            WindowSpec(observation=48, horizon=horizon),
        ],
        primary_metric="recall",
    )


def stock_direction_preset(
    *,
    price_col: str = "close",
    horizon: int = 1,
) -> DomainPreset:
    return DomainPreset(
        name="stock_direction",
        task_type="classification",
        target=TargetSpec(
            type="future_direction",
            column=price_col,
            horizon=horizon,
            name=f"{price_col}_direction_next_{horizon}",
        ),
        factors=FactorSpec(
            lag_columns=[price_col],
            lag_periods=[1, 2, 5, 10, 20],
            rolling_columns=[price_col],
            rolling_windows=[5, 10, 20],
            rolling_stats=["mean", "std", "min", "max"],
            diff_columns=[price_col],
            diff_periods=[1, 5, 20],
            ratio_to_rolling_mean_columns=[price_col],
            ratio_windows=[5, 20],
            calendar=["day_of_week", "month"],
        ),
        windows=[
            WindowSpec(observation=10, horizon=horizon),
            WindowSpec(observation=20, horizon=horizon),
            WindowSpec(observation=30, horizon=horizon),
            WindowSpec(observation=60, horizon=horizon),
        ],
        primary_metric="f1",
    )

