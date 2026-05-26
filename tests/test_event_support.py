import pandas as pd

from seqlens.automation import event_support_plan
from seqlens.experiments import EventExperimentConfig
from seqlens.factors import FactorSpec
from seqlens.targets import TargetSpec
from seqlens.windows import WindowSpec


def test_event_support_plan_scans_thresholds_and_horizons() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
            "entity": ["a"] * 30,
            "signal": [10 if hour % 10 in {5, 6, 7} else 0 for hour in range(30)],
        }
    )
    config = EventExperimentConfig(
        data_path="unused.csv",
        time_col="timestamp",
        entity_col="entity",
        target=TargetSpec(
            type="future_window_event",
            column="signal",
            horizon=3,
            aggregation="sum",
            threshold=20,
        ),
        factors=FactorSpec(),
        window=WindowSpec(observation=3, horizon=3),
    )

    support = event_support_plan(
        frame,
        config,
        thresholds=[20, 30],
        horizons=[3, 6],
    )

    assert {"threshold", "horizon", "positive_support", "support_status"}.issubset(
        support.columns
    )
    assert set(support["horizon"]) == {3, 6}
    assert set(support["threshold"]) == {20.0, 30.0}
    assert set(support["split"]) == {"train", "validation", "test"}
