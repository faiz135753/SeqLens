from __future__ import annotations

import pandas as pd


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

