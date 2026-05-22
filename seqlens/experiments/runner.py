from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from seqlens.data.loader import load_csv
from seqlens.data.splitting import time_based_split
from seqlens.evaluation.metrics import RegressionMetrics, regression_metrics
from seqlens.experiments.config import ExperimentConfig
from seqlens.models.baselines import baseline_forecast
from seqlens.reports import write_actual_vs_predicted_plot, write_baseline_report


@dataclass(frozen=True)
class BaselineRunResult:
    run_dir: Path
    validation_metrics: RegressionMetrics
    test_metrics: RegressionMetrics

    def summary(self) -> str:
        return (
            f"Run directory: {self.run_dir}\n\n"
            "Validation metrics\n"
            f"{self.validation_metrics.summary()}\n\n"
            "Test metrics\n"
            f"{self.test_metrics.summary()}"
        )


@dataclass(frozen=True)
class BaselineComparisonResult:
    output_dir: Path
    runs: list[BaselineRunResult]
    comparison_path: Path

    def summary(self) -> str:
        lines = [
            f"Comparison directory: {self.output_dir}",
            f"Comparison table: {self.comparison_path}",
            "",
            "Runs:",
        ]
        lines.extend(f"- {run.run_dir.name}" for run in self.runs)
        return "\n".join(lines)


def run_baseline_comparison(
    config: ExperimentConfig,
    *,
    output_dir: str | Path = "runs",
) -> BaselineComparisonResult:
    comparison_dir = _create_run_dir(output_dir, model_name="baseline_comparison")
    baseline_models = [model for model in config.models if model in {"naive", "moving_average"}]
    if not baseline_models:
        raise ValueError("No supported baseline models found in config.models.")

    runs: list[BaselineRunResult] = []
    rows: list[dict[str, float | str | None]] = []
    for model_name in baseline_models:
        run = run_baseline(config, model_name=model_name, output_dir=comparison_dir)
        runs.append(run)
        rows.append(
            {
                "model": model_name,
                "run_dir": run.run_dir.name,
                "validation_mae": run.validation_metrics.mae,
                "validation_rmse": run.validation_metrics.rmse,
                "validation_mape": run.validation_metrics.mape,
                "validation_direction_accuracy": run.validation_metrics.direction_accuracy,
                "test_mae": run.test_metrics.mae,
                "test_rmse": run.test_metrics.rmse,
                "test_mape": run.test_metrics.mape,
                "test_direction_accuracy": run.test_metrics.direction_accuracy,
            }
        )

    comparison_path = comparison_dir / "comparison.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)

    return BaselineComparisonResult(
        output_dir=comparison_dir,
        runs=runs,
        comparison_path=comparison_path,
    )


def run_naive_baseline(
    config: ExperimentConfig,
    *,
    output_dir: str | Path = "runs",
) -> BaselineRunResult:
    return run_baseline(config, model_name="naive", output_dir=output_dir)


def run_baseline(
    config: ExperimentConfig,
    *,
    model_name: str,
    output_dir: str | Path = "runs",
) -> BaselineRunResult:
    dataset = load_csv(config.data_path, config.time_col, config.target_col)
    split = time_based_split(
        dataset.frame,
        validation_size=config.validation_size,
        test_size=config.test_size,
    )

    validation_predictions = baseline_forecast(
        model_name,
        split.train[config.target_col],
        len(split.validation),
        moving_average_window=config.moving_average_window,
    )
    validation_previous = _previous_values(split.train, split.validation, config.target_col)
    validation_metrics = regression_metrics(
        split.validation[config.target_col],
        validation_predictions,
        previous_actual=validation_previous,
    )

    test_history = pd.concat(
        [split.train[config.target_col], split.validation[config.target_col]],
        ignore_index=True,
    )
    test_predictions = baseline_forecast(
        model_name,
        test_history,
        len(split.test),
        moving_average_window=config.moving_average_window,
    )
    test_previous = _previous_values(
        pd.concat([split.train, split.validation], ignore_index=True),
        split.test,
        config.target_col,
    )
    test_metrics = regression_metrics(
        split.test[config.target_col],
        test_predictions,
        previous_actual=test_previous,
    )

    run_dir = _create_run_dir(output_dir, model_name=model_name)
    _write_run_artifacts(
        run_dir=run_dir,
        model_name=model_name,
        config=config,
        dataset_shape=dataset.frame.shape,
        validation_frame=split.validation,
        validation_predictions=validation_predictions,
        validation_metrics=validation_metrics,
        test_frame=split.test,
        test_predictions=test_predictions,
        test_metrics=test_metrics,
        time_col=config.time_col,
        target_col=config.target_col,
    )

    return BaselineRunResult(
        run_dir=run_dir,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def _previous_values(
    history: pd.DataFrame,
    evaluation: pd.DataFrame,
    target_col: str,
) -> pd.Series:
    combined = pd.concat([history[[target_col]], evaluation[[target_col]]], ignore_index=True)
    return combined[target_col].shift(1).iloc[len(history) :].reset_index(drop=True)


def _create_run_dir(output_dir: str | Path, *, model_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{timestamp}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_run_artifacts(
    *,
    run_dir: Path,
    model_name: str,
    config: ExperimentConfig,
    dataset_shape: tuple[int, int],
    validation_frame: pd.DataFrame,
    validation_predictions: pd.Series,
    validation_metrics: RegressionMetrics,
    test_frame: pd.DataFrame,
    test_predictions: pd.Series,
    test_metrics: RegressionMetrics,
    time_col: str,
    target_col: str,
) -> None:
    config.to_yaml(run_dir / "config.yaml")

    metadata = {
        "model": model_name,
        "dataset_rows": dataset_shape[0],
        "dataset_columns": dataset_shape[1],
        "created_at": datetime.now(UTC).isoformat(),
    }
    with (run_dir / "metadata.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

    metrics = {
        "validation": validation_metrics.to_dict(),
        "test": test_metrics.to_dict(),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    validation_prediction_frame = _prediction_frame(
        validation_frame,
        validation_predictions,
        time_col=time_col,
        target_col=target_col,
    )
    test_prediction_frame = _prediction_frame(
        test_frame,
        test_predictions,
        time_col=time_col,
        target_col=target_col,
    )

    validation_prediction_frame.to_csv(run_dir / "validation_predictions.csv", index=False)
    test_prediction_frame.to_csv(run_dir / "test_predictions.csv", index=False)

    validation_plot = "validation_actual_vs_predicted.png"
    test_plot = "test_actual_vs_predicted.png"
    write_actual_vs_predicted_plot(
        validation_prediction_frame,
        path=run_dir / validation_plot,
        title="Validation Actual vs Predicted",
    )
    write_actual_vs_predicted_plot(
        test_prediction_frame,
        path=run_dir / test_plot,
        title="Test Actual vs Predicted",
    )
    write_baseline_report(
        path=run_dir / "report.md",
        model_name=model_name,
        config=config,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_plot=validation_plot,
        test_plot=test_plot,
    )


def _prediction_frame(
    frame: pd.DataFrame,
    predictions: pd.Series,
    *,
    time_col: str,
    target_col: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            time_col: frame[time_col].reset_index(drop=True),
            "actual": frame[target_col].reset_index(drop=True),
            "predicted": predictions.reset_index(drop=True),
        }
    )
    result["error"] = result["actual"] - result["predicted"]
    return result
