import pandas as pd

from seqlens.automation import recommend_event_factors
from seqlens.experiments import EventExperimentConfig
from seqlens.factors import FactorSpec
from seqlens.targets import TargetSpec
from seqlens.windows import WindowSpec


def test_recommend_event_factors_for_target_and_external_numeric_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
            "entity": ["a", "a", "a", "a"],
            "signal": [0, 1, 3, 8],
            "external": [10, 11, 11, 15],
            "label": ["x", "x", "y", "y"],
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
            threshold=5,
        ),
        factors=FactorSpec(),
        window=WindowSpec(observation=3, horizon=3),
    )

    recommendations = recommend_event_factors(frame, config)

    target_recommendations = [
        recommendation
        for recommendation in recommendations
        if recommendation.column == "signal"
    ]
    external_recommendations = [
        recommendation
        for recommendation in recommendations
        if recommendation.column == "external"
    ]
    assert len(target_recommendations) == 3
    assert len(external_recommendations) == 2
    assert all(recommendation.column != "entity" for recommendation in recommendations)
