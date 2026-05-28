from pathlib import Path

import pandas as pd

from seqlens.automation import station_level_metrics


def test_station_level_metrics_reads_candidate_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=4, freq="h"),
            "actual": [1, 0, 1, 0],
            "predicted": [1, 0, 0, 1],
            "entity": ["a", "a", "b", "b"],
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)
    leaderboard = pd.DataFrame(
        [
            {
                "candidate": "candidate-a",
                "threshold": 80,
                "observation": 12,
                "model": "lgbm",
                "threshold_strategy": "maximize_f1",
                "run_dir": str(run_dir),
            }
        ]
    )

    metrics = station_level_metrics(leaderboard)

    assert set(metrics["entity"]) == {"a", "b"}
    assert {"precision", "recall", "false_alarm_rate", "positive_support"}.issubset(
        metrics.columns
    )
