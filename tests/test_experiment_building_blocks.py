import pandas as pd

from seqlens.automation import generate_candidates
from seqlens.factors import FactorSpec, build_factors
from seqlens.presets import rainfall_extreme_rain_preset, stock_direction_preset
from seqlens.targets import TargetSpec, build_target
from seqlens.windows import WindowSpec, make_supervised_frame


def test_future_window_event_target() -> None:
    frame = pd.DataFrame({"rainfall": [0, 10, 30, 50, 0]})
    spec = TargetSpec(
        type="future_window_event",
        column="rainfall",
        horizon=3,
        aggregation="sum",
        threshold=80,
        name="heavy_rain_next_3h",
    )

    target = build_target(frame, spec)

    assert target.name == "heavy_rain_next_3h"
    assert target.iloc[0] == 1
    assert pd.isna(target.iloc[-1])


def test_factor_builder_creates_lag_rolling_ratio_and_calendar_factors() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "close": [10, 11, 12, 13, 14, 15],
        }
    )
    spec = FactorSpec(
        lag_columns=["close"],
        lag_periods=[1],
        rolling_columns=["close"],
        rolling_windows=[3],
        rolling_stats=["mean"],
        ratio_to_rolling_mean_columns=["close"],
        ratio_windows=[3],
        calendar=["day_of_week", "month"],
    )

    factors = build_factors(frame, time_col="date", spec=spec)

    assert "close_lag_1" in factors.columns
    assert "close_roll_3_mean" in factors.columns
    assert "close_ratio_roll_3_mean" in factors.columns
    assert "day_of_week" in factors.columns
    assert "month" in factors.columns


def test_window_slicer_builds_supervised_frame() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "close": [10, 11, 12, 13, 14, 15, 16, 17],
        }
    )

    supervised = make_supervised_frame(
        frame,
        time_col="date",
        target_spec=TargetSpec(type="future_direction", column="close", horizon=1),
        factor_spec=FactorSpec(
            lag_columns=["close"],
            lag_periods=[1, 2],
            rolling_columns=["close"],
            rolling_windows=[3],
            rolling_stats=["mean"],
        ),
        window=WindowSpec(observation=3, horizon=1),
    )

    assert "close_direction_t_plus_1" in supervised.columns
    assert "close_lag_1" in supervised.columns
    assert len(supervised) > 0


def test_domain_presets_generate_experiment_candidates() -> None:
    rainfall = rainfall_extreme_rain_preset(rain_col="rainfall", horizon=3, threshold=80)
    stock = stock_direction_preset(price_col="close", horizon=1)

    rainfall_candidates = generate_candidates(rainfall, models=["lgbm", "lstm"])
    stock_candidates = generate_candidates(stock, models=["lgbm"])

    assert rainfall_candidates[0].name.startswith("rainfall_extreme_rain_lgbm")
    assert len(rainfall_candidates) == 6
    assert len(stock_candidates) == 4
    assert stock_candidates[0].primary_metric == "f1"

