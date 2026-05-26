from pathlib import Path

import pandas as pd
import pytest

from seqlens.experiments import EventExperimentConfig, run_event_baseline, with_event_model


def test_run_event_baseline_for_multi_station_rainfall(tmp_path: Path) -> None:
    rows = []
    for station_id in ["A", "B"]:
        for hour in range(36):
            rows.append(
                {
                    "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour),
                    "station_id": station_id,
                    "rainfall": 30 if 10 <= hour <= 12 and station_id == "A" else 0,
                }
            )
    data_path = tmp_path / "rainfall.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    config_path = tmp_path / "rainfall.yaml"
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
""",
        encoding="utf-8",
    )

    config = EventExperimentConfig.from_yaml(config_path)
    result = run_event_baseline(config, output_dir=tmp_path)

    assert result.run_dir.exists()
    assert result.supervised_rows > 0
    assert result.event_rate > 0
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "report.md").exists()
    assert (result.run_dir / "validation_predictions.csv").exists()
    assert (result.run_dir / "test_predictions.csv").exists()


def test_run_lgbm_event_model_when_lightgbm_is_available(tmp_path: Path) -> None:
    try:
        __import__("lightgbm")
    except Exception as exc:
        pytest.skip(f"LightGBM is not available in this environment: {exc}")
    rows = []
    for hour in range(80):
        rows.append(
            {
                "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour),
                "station_id": "A",
                "rainfall": 30 if hour % 12 in {8, 9, 10} else 0,
            }
        )
    data_path = tmp_path / "rainfall.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    config_path = tmp_path / "rainfall.yaml"
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
""",
        encoding="utf-8",
    )

    config = with_event_model(EventExperimentConfig.from_yaml(config_path), "lgbm")
    result = run_event_baseline(config, output_dir=tmp_path)

    assert result.threshold is not None
    assert (result.run_dir / "feature_importance.csv").exists()
    predictions = pd.read_csv(result.run_dir / "test_predictions.csv")
    assert "probability" in predictions.columns
