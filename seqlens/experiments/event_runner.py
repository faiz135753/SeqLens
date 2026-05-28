from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from seqlens.data.splitting import time_based_split
from seqlens.evaluation import ClassificationMetrics, classification_metrics
from seqlens.factors import FactorSpec
from seqlens.models.lgbm import (
    lgbm_feature_importance,
    predict_lgbm_event_probability,
    train_lgbm_classifier,
)
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
    threshold_strategy: str = "maximize_f1"
    precision_floor: float = 0.05
    false_alarm_cap: float = 0.05
    event_baseline_column: str | None = None
    event_baseline_window: int | None = None
    event_baseline_aggregation: str = "sum"
    event_baseline_threshold: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EventExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        data = raw["data"]
        target = raw["target"]
        factors = raw["factors"]
        window = raw["window"]
        evaluation = raw.get("evaluation", {})
        threshold_strategy = evaluation.get("threshold_strategy", {})
        event_baseline = raw.get("event_baseline", {})
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
            threshold_strategy=threshold_strategy.get("name", "maximize_f1"),
            precision_floor=float(threshold_strategy.get("precision_floor", 0.05)),
            false_alarm_cap=float(threshold_strategy.get("false_alarm_cap", 0.05)),
            event_baseline_column=event_baseline.get("column", target["column"]),
            event_baseline_window=event_baseline.get("window"),
            event_baseline_aggregation=event_baseline.get("aggregation", "sum"),
            event_baseline_threshold=event_baseline.get("threshold", target.get("threshold")),
        )


@dataclass(frozen=True)
class EventRunResult:
    run_dir: Path
    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    supervised_rows: int
    event_rate: float
    threshold: float | None = None

    def summary(self) -> str:
        threshold = "n/a" if self.threshold is None else f"{self.threshold:.3f}"
        return (
            f"Run directory: {self.run_dir}\n"
            f"Supervised rows: {self.supervised_rows}\n"
            f"Overall event rate: {self.event_rate:.3f}\n\n"
            f"Decision threshold: {threshold}\n\n"
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
    supervised = make_event_supervised_dataset(frame, config)
    split = time_based_split(
        supervised,
        validation_size=config.validation_size,
        test_size=config.test_size,
    )

    target_col = config.target.output_name
    feature_importance = None
    validation_probabilities = None
    test_probabilities = None
    threshold = None

    if config.model == "lgbm":
        feature_columns = _feature_columns(supervised, config=config, target_col=target_col)
        model = train_lgbm_classifier(
            split.train,
            target_col=target_col,
            feature_columns=feature_columns,
        )
        validation_probabilities = predict_lgbm_event_probability(model, split.validation)
        threshold = _choose_threshold(
            split.validation[target_col],
            validation_probabilities,
            strategy=config.threshold_strategy,
            precision_floor=config.precision_floor,
            false_alarm_cap=config.false_alarm_cap,
        )
        validation_predictions = (validation_probabilities >= threshold).astype("Int64")

        train_validation = pd.concat([split.train, split.validation], ignore_index=True)
        final_model = train_lgbm_classifier(
            train_validation,
            target_col=target_col,
            feature_columns=feature_columns,
        )
        test_probabilities = predict_lgbm_event_probability(final_model, split.test)
        test_predictions = (test_probabilities >= threshold).astype("Int64")
        feature_importance = lgbm_feature_importance(final_model)
    elif config.model == "recent_window_threshold":
        validation_predictions = _recent_window_threshold_predict(split.validation, config)
        test_predictions = _recent_window_threshold_predict(split.test, config)
    else:
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
        validation_probabilities=validation_probabilities,
        validation_metrics=validation_metrics,
        test_frame=split.test,
        test_predictions=test_predictions,
        test_probabilities=test_probabilities,
        test_metrics=test_metrics,
        feature_importance=feature_importance,
        threshold=threshold,
        target_col=target_col,
    )

    return EventRunResult(
        run_dir=run_dir,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        supervised_rows=len(supervised),
        event_rate=float((supervised[target_col] == 1).mean()),
        threshold=threshold,
    )


def with_event_model(config: EventExperimentConfig, model: str) -> EventExperimentConfig:
    return replace(config, model=model)


def with_threshold_strategy(
    config: EventExperimentConfig,
    strategy: str,
) -> EventExperimentConfig:
    return replace(config, threshold_strategy=strategy)


def with_event_threshold(config: EventExperimentConfig, threshold: float) -> EventExperimentConfig:
    return replace(
        config,
        target=replace(
            config.target,
            threshold=threshold,
            name=f"{config.target.column}_{config.target.aggregation}_next_"
            f"{config.target.horizon}_ge_{_format_threshold(threshold)}",
        ),
        event_baseline_threshold=threshold,
    )


def with_observation_window(
    config: EventExperimentConfig,
    observation: int,
) -> EventExperimentConfig:
    return replace(config, window=replace(config.window, observation=observation))


def _format_threshold(threshold: float) -> str:
    if threshold == int(threshold):
        return str(int(threshold))
    return str(threshold).replace(".", "p")


def make_event_supervised_dataset(
    frame: pd.DataFrame,
    config: EventExperimentConfig,
) -> pd.DataFrame:
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


def _recent_window_threshold_predict(
    frame: pd.DataFrame,
    config: EventExperimentConfig,
) -> pd.Series:
    if config.event_baseline_column is None:
        raise ValueError("recent_window_threshold requires event_baseline.column.")
    if config.event_baseline_window is None:
        raise ValueError("recent_window_threshold requires event_baseline.window.")
    if config.event_baseline_threshold is None:
        raise ValueError("recent_window_threshold requires event_baseline.threshold.")

    factor_col = (
        f"{config.event_baseline_column}_roll_{config.event_baseline_window}_"
        f"{config.event_baseline_aggregation}"
    )
    if factor_col not in frame.columns:
        raise ValueError(
            f"recent_window_threshold requires factor column `{factor_col}`. "
            "Add the matching rolling factor to config."
        )
    predictions = frame[factor_col] >= config.event_baseline_threshold
    return predictions.astype("Int64").reset_index(drop=True)


def _feature_columns(
    supervised: pd.DataFrame,
    *,
    config: EventExperimentConfig,
    target_col: str,
) -> list[str]:
    excluded = {config.time_col, target_col}
    if config.entity_col:
        excluded.add(config.entity_col)
    feature_columns = [column for column in supervised.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No feature columns available for LGBM.")
    return feature_columns


def _choose_threshold(
    actual: pd.Series,
    probabilities: pd.Series,
    *,
    strategy: str,
    precision_floor: float,
    false_alarm_cap: float,
) -> float:
    best_threshold = 0.5
    best_score: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    for index in range(5, 96, 5):
        threshold = index / 100
        predicted = (probabilities >= threshold).astype(int)
        metrics = classification_metrics(actual, predicted)
        score = _threshold_score(
            metrics,
            strategy=strategy,
            precision_floor=precision_floor,
            false_alarm_cap=false_alarm_cap,
        )
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return best_threshold


def _threshold_score(
    metrics: ClassificationMetrics,
    *,
    strategy: str,
    precision_floor: float,
    false_alarm_cap: float,
) -> tuple[float, float, float]:
    if strategy == "maximize_f1":
        return (metrics.f1, metrics.recall, -metrics.false_alarm_rate)
    if strategy == "maximize_recall":
        return (metrics.recall, metrics.f1, -metrics.false_alarm_rate)
    if strategy == "maximize_recall_with_precision_floor":
        if metrics.precision < precision_floor:
            return (-1.0, metrics.recall, -metrics.false_alarm_rate)
        return (metrics.recall, metrics.f1, -metrics.false_alarm_rate)
    if strategy == "minimize_miss_rate_with_false_alarm_cap":
        if metrics.false_alarm_rate > false_alarm_cap:
            return (-1.0, -metrics.miss_rate, metrics.precision)
        return (-metrics.miss_rate, metrics.precision, metrics.f1)
    raise ValueError(f"Unsupported threshold strategy: {strategy}")


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
    validation_probabilities: pd.Series | None,
    validation_metrics: ClassificationMetrics,
    test_frame: pd.DataFrame,
    test_predictions: pd.Series,
    test_probabilities: pd.Series | None,
    test_metrics: ClassificationMetrics,
    feature_importance: pd.DataFrame | None,
    threshold: float | None,
    target_col: str,
) -> None:
    metadata = {
        "model": config.model,
        "task_type": "classification",
        "target": target_col,
        "supervised_rows": len(supervised),
        "event_rate": float((supervised[target_col] == 1).mean()),
        "decision_threshold": threshold,
        "threshold_strategy": config.threshold_strategy,
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
    _event_predictions(
        validation_frame,
        validation_predictions,
        target_col,
        entity_col=config.entity_col,
        probabilities=validation_probabilities,
    ).to_csv(
        run_dir / "validation_predictions.csv",
        index=False,
    )
    _event_predictions(
        test_frame,
        test_predictions,
        target_col,
        entity_col=config.entity_col,
        probabilities=test_probabilities,
    ).to_csv(
        run_dir / "test_predictions.csv",
        index=False,
    )
    if feature_importance is not None:
        feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)
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
    *,
    entity_col: str | None = None,
    probabilities: pd.Series | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "time": frame.iloc[:, 0].reset_index(drop=True),
            "actual": frame[target_col].reset_index(drop=True).astype(int),
            "predicted": predictions.reset_index(drop=True).astype(int),
        }
    )
    if entity_col and entity_col in frame.columns:
        result["entity"] = frame[entity_col].reset_index(drop=True)
    if probabilities is not None:
        result["probability"] = probabilities.reset_index(drop=True)
    return result


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
| Decision threshold | {metadata["decision_threshold"] if metadata["decision_threshold"] is not None else "n/a"} |
| Threshold strategy | {metadata["threshold_strategy"]} |

## Validation Metrics

{_classification_table(validation_metrics)}

## Test Metrics

{_classification_table(test_metrics)}

## Interpretation

Event baselines are sanity checks before promoting a task to learned models. If a baseline achieves high accuracy but zero recall, the event is likely rare and accuracy is not a useful primary metric.
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
