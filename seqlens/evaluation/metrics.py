from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float | None
    direction_accuracy: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)

    def summary(self) -> str:
        mape = "n/a" if self.mape is None else f"{self.mape:.3f}"
        direction = (
            "n/a" if self.direction_accuracy is None else f"{self.direction_accuracy:.3f}"
        )
        return (
            f"MAE: {self.mae:.3f}\n"
            f"RMSE: {self.rmse:.3f}\n"
            f"MAPE: {mape}\n"
            f"Direction accuracy: {direction}"
        )


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    miss_rate: float
    event_rate: float
    support: int
    positive_support: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.3f}\n"
            f"Precision: {self.precision:.3f}\n"
            f"Recall: {self.recall:.3f}\n"
            f"F1: {self.f1:.3f}\n"
            f"False alarm rate: {self.false_alarm_rate:.3f}\n"
            f"Miss rate: {self.miss_rate:.3f}\n"
            f"Event rate: {self.event_rate:.3f}\n"
            f"Support: {self.support}\n"
            f"Positive support: {self.positive_support}"
        )


def regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    previous_actual: pd.Series | np.ndarray | None = None,
) -> RegressionMetrics:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if len(actual_array) != len(predicted_array):
        raise ValueError("actual and predicted must have the same length.")
    if len(actual_array) == 0:
        raise ValueError("Cannot calculate metrics for empty arrays.")

    error = actual_array - predicted_array
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    mape = _mape(actual_array, predicted_array)
    direction_accuracy = _direction_accuracy(actual_array, predicted_array, previous_actual)

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        direction_accuracy=direction_accuracy,
    )


def classification_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> ClassificationMetrics:
    actual_array = np.asarray(actual, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)
    if len(actual_array) != len(predicted_array):
        raise ValueError("actual and predicted must have the same length.")
    if len(actual_array) == 0:
        raise ValueError("Cannot calculate metrics for empty arrays.")

    tp = int(np.sum((actual_array == 1) & (predicted_array == 1)))
    tn = int(np.sum((actual_array == 0) & (predicted_array == 0)))
    fp = int(np.sum((actual_array == 0) & (predicted_array == 1)))
    fn = int(np.sum((actual_array == 1) & (predicted_array == 0)))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    false_alarm_rate = _safe_divide(fp, fp + tn)
    miss_rate = _safe_divide(fn, fn + tp)
    support = int(len(actual_array))
    positive_support = int(np.sum(actual_array == 1))

    return ClassificationMetrics(
        accuracy=_safe_divide(tp + tn, support),
        precision=precision,
        recall=recall,
        f1=f1,
        false_alarm_rate=false_alarm_rate,
        miss_rate=miss_rate,
        event_rate=_safe_divide(positive_support, support),
        support=support,
        positive_support=positive_support,
    )


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    nonzero = actual != 0
    if not np.any(nonzero):
        return None
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


def _direction_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    previous_actual: pd.Series | np.ndarray | None,
) -> float | None:
    if previous_actual is None:
        return None

    previous_array = np.asarray(previous_actual, dtype=float)
    if len(previous_array) != len(actual):
        raise ValueError("previous_actual must have the same length as actual.")

    actual_direction = np.sign(actual - previous_array)
    predicted_direction = np.sign(predicted - previous_array)
    comparable = actual_direction != 0
    if not np.any(comparable):
        return None
    return float(np.mean(actual_direction[comparable] == predicted_direction[comparable]))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
