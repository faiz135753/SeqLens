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

