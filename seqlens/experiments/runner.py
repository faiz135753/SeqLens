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
from seqlens.models.baselines import naive_forecast


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


def run_naive_baseline(
    config: ExperimentConfig,
    *,
    output_dir: str | Path = "runs",
) -> BaselineRunResult:
    dataset = load_csv(config.data_path, config.time_col, config.target_col)
    split = time_based_split(
        dataset.frame,
        validation_size=config.validation_size,
        test_size=config.test_size,
    )

    validation_predictions = naive_forecast(
        split.train[config.target_col],
        len(split.validation),
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
    test_predictions = naive_forecast(test_history, len(split.test))
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

    run_dir = _create_run_dir(output_dir, model_name="naive")
    _write_run_artifacts(
        run_dir=run_dir,
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
        "model": "naive",
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

    _prediction_frame(
        validation_frame,
        validation_predictions,
        time_col=time_col,
        target_col=target_col,
    ).to_csv(run_dir / "validation_predictions.csv", index=False)
    _prediction_frame(
        test_frame,
        test_predictions,
        time_col=time_col,
        target_col=target_col,
    ).to_csv(run_dir / "test_predictions.csv", index=False)


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

