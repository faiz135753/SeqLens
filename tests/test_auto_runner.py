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
  baselines: [event_majority, recent_window_threshold]
event_baseline:
  type: recent_window_threshold
  column: rainfall
  window: 3
  aggregation: sum
  threshold: 80
evaluation:
  primary_metric: recall
automation:
  thresholds: [60, 80]
  horizons: [3, 6]
  observation_windows: [6]
  models: [event_majority, recent_window_threshold]
  threshold_strategies: [maximize_f1, maximize_recall]
""",
        encoding="utf-8",
    )

    result = run_auto_event_experiment(config_path, output_dir=tmp_path)

    assert result.candidate_count == 4
    assert result.leaderboard_path.exists()
    assert result.distribution_path.exists()
    assert result.support_plan_path.exists()
    assert result.extreme_profile_path.exists()
    assert result.diagnosis_path.exists()
    assert result.factor_recommendations_path.exists()
    assert result.recommendations_path.exists()
    assert result.report_path.exists()

    leaderboard = pd.read_csv(result.leaderboard_path)
    assert len(leaderboard) == 4
    assert {"candidate", "threshold", "model", "validation_recall"}.issubset(
        leaderboard.columns
    )
    assert "threshold_strategy" in leaderboard.columns
    assert "extreme_rarity_level" in leaderboard.columns

    distribution = pd.read_csv(result.distribution_path)
    assert {"split", "entity", "positive_support", "event_rate"}.issubset(
        distribution.columns
    )
    assert set(distribution["split"]) == {"train", "validation", "test"}

    support_plan = pd.read_csv(result.support_plan_path)
    assert {"threshold", "horizon", "support_status"}.issubset(support_plan.columns)
    assert set(support_plan["horizon"]) == {3, 6}
    assert (result.run_dir / "event_support_plan.md").exists()

    extreme_profile = pd.read_csv(result.extreme_profile_path)
    assert {"rarity_level", "promotion_gate", "imbalance_ratio"}.issubset(
        extreme_profile.columns
    )
    assert (result.run_dir / "extreme_profile.md").exists()

    diagnosis = pd.read_csv(result.diagnosis_path)
    assert {"stage", "status", "reason"}.issubset(diagnosis.columns)
    assert {"lgbm", "lstm"}.issubset(set(diagnosis["stage"]))
    assert (result.run_dir / "experiment_diagnosis.md").exists()

    factor_recommendations = pd.read_csv(result.factor_recommendations_path)
    assert {"column", "factor_type", "windows", "parameters"}.issubset(
        factor_recommendations.columns
    )
    assert "rainfall" in set(factor_recommendations["column"])
    assert (result.run_dir / "factor_recommendations.md").exists()
