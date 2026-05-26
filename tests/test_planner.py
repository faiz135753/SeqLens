import pandas as pd

from seqlens.automation import diagnose_auto_experiment


def test_diagnose_auto_experiment_blocks_lstm_without_lgbm() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "model": "recent_window_threshold",
                "validation_positive_support": 40,
                "test_recall": 0.10,
                "test_f1": 0.10,
                "test_false_alarm_rate": 0.01,
                "validation_recall": 0.12,
            }
        ]
    )
    distribution = pd.DataFrame(
        [
            {"split": "validation", "entity": "all", "positive_support": 40},
            {"split": "test", "entity": "all", "positive_support": 35},
        ]
    )

    diagnosis = diagnose_auto_experiment(leaderboard, distribution)

    assert diagnosis.can_promote_to_lgbm
    assert not diagnosis.can_promote_to_lstm


def test_diagnose_auto_experiment_promotes_lstm_after_stable_lgbm() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "model": "recent_window_threshold",
                "validation_positive_support": 50,
                "test_recall": 0.08,
                "test_f1": 0.08,
                "test_false_alarm_rate": 0.02,
                "validation_recall": 0.09,
            },
            {
                "model": "lgbm",
                "validation_positive_support": 50,
                "validation_recall": 0.30,
                "test_recall": 0.24,
                "test_f1": 0.12,
                "test_false_alarm_rate": 0.08,
            },
        ]
    )
    distribution = pd.DataFrame(
        [
            {"split": "validation", "entity": "all", "positive_support": 50},
            {"split": "test", "entity": "all", "positive_support": 45},
        ]
    )

    diagnosis = diagnose_auto_experiment(leaderboard, distribution)

    assert diagnosis.can_promote_to_lgbm
    assert diagnosis.can_promote_to_lstm
