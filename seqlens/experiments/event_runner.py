from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from seqlens.data.splitting import time_based_split
from seqlens.evaluation import ClassificationMetrics, classification_metrics
from seqlens.factors import FactorSpec
from seqlens.models import event_majority_forecast
from seqlens.targets import TargetSpec
from seqlens.windows import WindowSpec, make_supervised_frame


@dataclass(frozen=True)
class EventExperimentConfig:
    data_path: str
    time_col: str
    target: TargetSpec
    factors: FactorSpec
    window: WindowSpec
    entity_col: str | None = None
    validation_size: float = 0.2
    test_size: float = 0.2
    model: str = "event_majority"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EventExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        data = raw["data"]
        target = raw["target"]
        factors = raw["factors"]
        window = raw["window"]
        observation_windows = window.get("observation_windows") or [window["observation"]]
        horizons = window.get("horizons") or [target["horizon"]]

        return cls(
            data_path=data["path"],
            time_col=data["time_col"],
            entity_col=data.get("entity_col"),
            target=TargetSpec(
                type=target["type"],
                column=target["column"],
                horizon=int(target["horizon"]),
                aggregation=target.get("aggregation", "value"),
                threshold=target.get("threshold"),
                name=target.get("name"),
            ),
            factors=_factor_spec_from_yaml(factors),
            window=WindowSpec(
                observation=int(observation_windows[0]),
                horizon=int(horizons[0]),
                step=int(window.get("step", 1)),
            ),
            model=(raw.get("models", {}).get("baselines") or ["event_majority"])[0],
        )


@dataclass(frozen=True)
class EventRunResult:
    run_dir: Path
    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    supervised_rows: int
    event_rate: float

    def summary(self) -> str:
        return (
            f"Run directory: {self.run_dir}\n"
            f"Supervised rows: {self.supervised_rows}\n"
            f"Overall event rate: {self.event_rate:.3f}\n\n"
            "Validation metrics\n"
            f"{self.validation_metrics.summary()}\n\n"
            "Test metrics\n"
            f"{self.test_metrics.summary()}"
        )


def run_event_baseline(
    config: EventExperimentConfig,
    *,
    output_dir: str | Path = "runs",
) -> EventRunResult:
    frame = pd.read_csv(config.data_path)
    supervised = _make_supervised_dataset(frame, config)
    split = time_based_split(
        supervised,
        validation_size=config.validation_size,
        test_size=config.test_size,
    )

    target_col = config.target.output_name
    validation_predictions = _event_baseline_predict(
        config.model,
        split.train[target_col],
        len(split.validation),
    )
    test_history = pd.concat([split.train[target_col], split.validation[target_col]])
    test_predictions = _event_baseline_predict(config.model, test_history, len(split.test))

    validation_metrics = classification_metrics(split.validation[target_col], validation_predictions)
    test_metrics = classification_metrics(split.test[target_col], test_predictions)

    run_dir = _create_run_dir(output_dir, model_name=config.model)
    _write_event_artifacts(
        run_dir=run_dir,
        config=config,
        supervised=supervised,
        validation_frame=split.validation,
        validation_predictions=validation_predictions,
        validation_metrics=validation_metrics,
        test_frame=split.test,
        test_predictions=test_predictions,
        test_metrics=test_metrics,
        target_col=target_col,
    )

    return EventRunResult(
        run_dir=run_dir,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        supervised_rows=len(supervised),
        event_rate=float((supervised[target_col] == 1).mean()),
    )


def _make_supervised_dataset(frame: pd.DataFrame, config: EventExperimentConfig) -> pd.DataFrame:
    if config.entity_col and config.entity_col in frame.columns:
        pieces = []
        for entity_value, group in frame.groupby(config.entity_col, sort=False):
            supervised = make_supervised_frame(
                group,
                time_col=config.time_col,
                target_spec=config.target,
                factor_spec=config.factors,
                window=config.window,
            )
            supervised[config.entity_col] = entity_value
            pieces.append(supervised)
        if not pieces:
            raise ValueError("No supervised rows could be created.")
        result = pd.concat(pieces, ignore_index=True)
        return result.sort_values(config.time_col).reset_index(drop=True)

    return make_supervised_frame(
        frame,
        time_col=config.time_col,
        target_spec=config.target,
        factor_spec=config.factors,
        window=config.window,
    )


def _event_baseline_predict(model: str, history: pd.Series, horizon: int) -> pd.Series:
    if model in {"event_majority", "event_naive"}:
        return event_majority_forecast(history, horizon)
    raise ValueError(f"Unsupported event baseline model: {model}")


def _factor_spec_from_yaml(raw: dict) -> FactorSpec:
    lag = raw.get("lag", {})
    rolling = raw.get("rolling", {})
    diff = raw.get("diff", {})
    ratio = raw.get("ratio_to_rolling_mean", {})
    return FactorSpec(
        lag_columns=lag.get("columns", []),
        lag_periods=lag.get("periods", []),
        rolling_columns=rolling.get("columns", []),
        rolling_windows=rolling.get("windows", []),
        rolling_stats=rolling.get("stats", ["mean"]),
        diff_columns=diff.get("columns", []),
        diff_periods=diff.get("periods", []),
        ratio_to_rolling_mean_columns=ratio.get("columns", []),
        ratio_windows=ratio.get("windows", []),
        calendar=raw.get("calendar", []),
    )


def _create_run_dir(output_dir: str | Path, *, model_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_dir)
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index}"
        run_dir = base_dir / f"{timestamp}_{model_name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique run directory under {base_dir}")


def _write_event_artifacts(
    *,
    run_dir: Path,
    config: EventExperimentConfig,
    supervised: pd.DataFrame,
    validation_frame: pd.DataFrame,
    validation_predictions: pd.Series,
    validation_metrics: ClassificationMetrics,
    test_frame: pd.DataFrame,
    test_predictions: pd.Series,
    test_metrics: ClassificationMetrics,
    target_col: str,
) -> None:
    metadata = {
        "model": config.model,
        "task_type": "classification",
        "target": target_col,
        "supervised_rows": len(supervised),
        "event_rate": float((supervised[target_col] == 1).mean()),
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

    supervised.head(5000).to_csv(run_dir / "supervised_preview.csv", index=False)
    _event_predictions(validation_frame, validation_predictions, target_col).to_csv(
        run_dir / "validation_predictions.csv",
        index=False,
    )
    _event_predictions(test_frame, test_predictions, target_col).to_csv(
        run_dir / "test_predictions.csv",
        index=False,
    )
    (run_dir / "report.md").write_text(
        _event_report_text(
            metadata=metadata,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        ),
        encoding="utf-8",
    )


def _event_predictions(
    frame: pd.DataFrame,
    predictions: pd.Series,
    target_col: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": frame.iloc[:, 0].reset_index(drop=True),
            "actual": frame[target_col].reset_index(drop=True).astype(int),
            "predicted": predictions.reset_index(drop=True).astype(int),
        }
    )


def _event_report_text(
    *,
    metadata: dict,
    validation_metrics: ClassificationMetrics,
    test_metrics: ClassificationMetrics,
) -> str:
    return f"""# SeqLens Event Baseline Report

## Experiment

| Field | Value |
|---|---:|
| Model | {metadata["model"]} |
| Target | {metadata["target"]} |
| Supervised rows | {metadata["supervised_rows"]} |
| Event rate | {metadata["event_rate"]:.3f} |

## Validation Metrics

{_classification_table(validation_metrics)}

## Test Metrics

{_classification_table(test_metrics)}

## Interpretation

This event baseline predicts the majority event class observed in the training history. If it achieves high accuracy but zero recall, the event is likely rare and accuracy is not a useful primary metric.
"""


def _classification_table(metrics: ClassificationMetrics) -> str:
    return f"""| Metric | Value |
|---|---:|
| Accuracy | {metrics.accuracy:.3f} |
| Precision | {metrics.precision:.3f} |
| Recall | {metrics.recall:.3f} |
| F1 | {metrics.f1:.3f} |
| False alarm rate | {metrics.false_alarm_rate:.3f} |
| Miss rate | {metrics.miss_rate:.3f} |
| Event rate | {metrics.event_rate:.3f} |
| Support | {metrics.support} |
| Positive support | {metrics.positive_support} |
"""
