import pandas as pd

from seqlens.automation import recommend_external_signals


def test_recommend_external_signals_when_baseline_beats_lgbm() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "model": "recent_window_threshold",
                "test_f1": 0.16,
                "test_recall": 0.16,
                "test_false_alarm_rate": 0.005,
            },
            {
                "model": "lgbm",
                "test_f1": 0.04,
                "test_recall": 0.27,
                "test_false_alarm_rate": 0.06,
            },
        ]
    )
    imbalance = pd.DataFrame(
        [
            {
                "threshold": 80,
                "horizon": 12,
                "severity": "imbalanced",
            }
        ]
    )

    recommendations = recommend_external_signals(
        leaderboard,
        imbalance,
        domain="rainfall",
    )

    groups = {recommendation.suggested_signal_group for recommendation in recommendations}
    assert "same_station_weather" in groups
    assert "spatial_rainfall_context" in groups
    assert recommendations[0].priority == 1


def test_recommend_external_signals_stays_empty_when_lgbm_clearly_improves() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "model": "recent_window_threshold",
                "test_f1": 0.10,
                "test_recall": 0.10,
                "test_false_alarm_rate": 0.005,
            },
            {
                "model": "lgbm",
                "test_f1": 0.30,
                "test_recall": 0.40,
                "test_false_alarm_rate": 0.02,
            },
        ]
    )
    imbalance = pd.DataFrame(
        [
            {
                "threshold": 80,
                "horizon": 12,
                "severity": "imbalanced",
            }
        ]
    )

    recommendations = recommend_external_signals(leaderboard, imbalance)

    assert recommendations == []
