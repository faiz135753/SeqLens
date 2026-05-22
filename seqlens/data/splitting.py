from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def time_based_split(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.2,
    test_size: float = 0.2,
) -> TimeSeriesSplit:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1.")

    row_count = len(frame)
    if row_count < 5:
        raise ValueError("At least 5 rows are required for train/validation/test splitting.")

    test_count = max(1, int(round(row_count * test_size)))
    validation_count = max(1, int(round(row_count * validation_size)))
    train_count = row_count - validation_count - test_count

    if train_count < 1:
        raise ValueError("Split sizes leave no rows for training.")

    train = frame.iloc[:train_count].copy()
    validation = frame.iloc[train_count : train_count + validation_count].copy()
    test = frame.iloc[train_count + validation_count :].copy()

    return TimeSeriesSplit(train=train, validation=validation, test=test)

