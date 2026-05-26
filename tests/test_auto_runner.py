from pathlib import Path

import pandas as pd

from seqlens.automation import run_auto_event_experiment


def test_auto_event_experiment_writes_leaderboard_and_reports(tmp_path: Path) -> None:
    rows = []
    for hour in range(72):
        rows.append(
            {
                "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour),
                "station_id": "A",
                "rainfall": 30 if hour % 18 in {10, 11, 12} else 0,
            }
        )
    data_path = tmp_path / "rainfall.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    config_path = tmp_path / "auto.yaml"
    config_path.write_text(
        f"""
domain: rainfall
task_type: classification
data:
  path: {data_path}
  time_col: datetime
  entity_col: station_id
target:
  type: future_window_event
  column: rainfall
  horizon: 3
  aggregation: sum
  threshold: 80
  name: heavy_rain_next_3h
window:
  observation_windows: [6]
  horizons: [3]
  step: 1
factors:
  lag:
    columns: [rainfall]
    periods: [1, 2, 3]
  rolling:
    columns: [rainfall]
    windows: [3]
    stats: [sum, max]
  calendar: [hour]
models:
  baselines: [event_majority]
evaluation:
  primary_metric: recall
automation:
  thresholds: [60, 80]
  observation_windows: [6]
  models: [event_majority]
  threshold_strategies: [maximize_f1, maximize_recall]
""",
        encoding="utf-8",
    )

    result = run_auto_event_experiment(config_path, output_dir=tmp_path)

    assert result.candidate_count == 2
    assert result.leaderboard_path.exists()
    assert result.distribution_path.exists()
    assert result.recommendations_path.exists()
    assert result.report_path.exists()

    leaderboard = pd.read_csv(result.leaderboard_path)
    assert len(leaderboard) == 2
    assert {"candidate", "threshold", "model", "validation_recall"}.issubset(
        leaderboard.columns
    )
    assert "threshold_strategy" in leaderboard.columns

    distribution = pd.read_csv(result.distribution_path)
    assert {"split", "entity", "positive_support", "event_rate"}.issubset(
        distribution.columns
    )
    assert set(distribution["split"]) == {"train", "validation", "test"}
