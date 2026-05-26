import pandas as pd

from seqlens.automation import ExtremeEventPolicy, assess_extreme_event_layer


def test_extreme_event_layer_marks_sparse_definitions() -> None:
    distribution = pd.DataFrame(
        [
            {
                "threshold": 80.0,
                "observation": 12,
                "split": "train",
                "entity": "A",
                "support": 1000,
                "positive_support": 8,
                "event_rate": 0.008,
            },
            {
                "threshold": 80.0,
                "observation": 12,
                "split": "validation",
                "entity": "A",
                "support": 300,
                "positive_support": 1,
                "event_rate": 0.0033,
            },
            {
                "threshold": 80.0,
                "observation": 12,
                "split": "test",
                "entity": "A",
                "support": 300,
                "positive_support": 2,
                "event_rate": 0.0067,
            },
        ]
    )

    profile = assess_extreme_event_layer(distribution)

    assert len(profile) == 1
    row = profile.iloc[0]
    assert row["rarity_level"] == "extreme_sparse"
    assert row["promotion_gate"] == "blocked"
    assert row["min_split_positive_support"] == 1


def test_extreme_event_layer_can_be_disabled() -> None:
    distribution = pd.DataFrame(
        [
            {
                "threshold": 40.0,
                "observation": 12,
                "split": "train",
                "entity": "all",
                "support": 100,
                "positive_support": 20,
                "event_rate": 0.2,
            }
        ]
    )

    profile = assess_extreme_event_layer(
        distribution,
        policy=ExtremeEventPolicy(enabled=False),
    )

    assert profile.empty
