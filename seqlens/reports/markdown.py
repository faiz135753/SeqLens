from __future__ import annotations

from pathlib import Path

from seqlens.evaluation.metrics import RegressionMetrics
from seqlens.experiments.config import ExperimentConfig


def write_baseline_report(
    *,
    path: str | Path,
    config: ExperimentConfig,
    validation_metrics: RegressionMetrics,
    test_metrics: RegressionMetrics,
    validation_plot: str,
    test_plot: str,
) -> None:
    report = _baseline_report_text(
        config=config,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_plot=validation_plot,
        test_plot=test_plot,
    )
    Path(path).write_text(report, encoding="utf-8")


def _baseline_report_text(
    *,
    config: ExperimentConfig,
    validation_metrics: RegressionMetrics,
    test_metrics: RegressionMetrics,
    validation_plot: str,
    test_plot: str,
) -> str:
    return f"""# SeqLens Baseline Report

## Experiment

| Field | Value |
|---|---|
| Model | naive |
| Data | `{config.data_path}` |
| Time column | `{config.time_col}` |
| Target column | `{config.target_col}` |
| Prediction horizon | {config.prediction_horizon} |
| Validation size | {config.validation_size} |
| Test size | {config.test_size} |
| Seed | {config.seed} |

## Validation Metrics

{_metrics_table(validation_metrics)}

![Validation actual vs predicted]({validation_plot})

## Test Metrics

{_metrics_table(test_metrics)}

![Test actual vs predicted]({test_plot})

## Interpretation

The naive baseline predicts future values using the latest observed target value from the available history. It is intentionally simple and should be treated as the minimum benchmark that more complex models, including LSTM, must beat.

Use validation metrics when tuning future experiments. Keep test metrics for final reporting so the experiment does not overfit the test split.
"""


def _metrics_table(metrics: RegressionMetrics) -> str:
    mape = "n/a" if metrics.mape is None else f"{metrics.mape:.3f}"
    direction_accuracy = (
        "n/a" if metrics.direction_accuracy is None else f"{metrics.direction_accuracy:.3f}"
    )
    return f"""| Metric | Value |
|---|---:|
| MAE | {metrics.mae:.3f} |
| RMSE | {metrics.rmse:.3f} |
| MAPE | {mape} |
| Direction Accuracy | {direction_accuracy} |
"""

