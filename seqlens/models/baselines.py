from __future__ import annotations

import pandas as pd


SUPPORTED_BASELINES = ("naive", "moving_average")


def naive_forecast(
    history: pd.Series,
    horizon: int,
) -> pd.Series:
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    clean_history = history.dropna()
    if clean_history.empty:
        raise ValueError("Naive forecast requires at least one non-missing historical value.")

    last_value = float(clean_history.iloc[-1])
    return pd.Series([last_value] * horizon)


def moving_average_forecast(
    history: pd.Series,
    horizon: int,
    *,
    window: int = 3,
) -> pd.Series:
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if window < 1:
        raise ValueError("window must be at least 1.")

    clean_history = history.dropna()
    if clean_history.empty:
        raise ValueError("Moving average forecast requires at least one non-missing value.")

    window_values = clean_history.tail(window)
    forecast_value = float(window_values.mean())
    return pd.Series([forecast_value] * horizon)


def baseline_forecast(
    model_name: str,
    history: pd.Series,
    horizon: int,
    *,
    moving_average_window: int = 3,
) -> pd.Series:
    if model_name == "naive":
        return naive_forecast(history, horizon)
    if model_name == "moving_average":
        return moving_average_forecast(
            history,
            horizon,
            window=moving_average_window,
        )
    supported = ", ".join(SUPPORTED_BASELINES)
    raise ValueError(f"Unsupported baseline model: {model_name}. Supported models: {supported}")
