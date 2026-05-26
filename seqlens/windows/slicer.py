from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from seqlens.factors import FactorSpec, build_factors
from seqlens.targets import TargetSpec, build_target


@dataclass(frozen=True)
class WindowSpec:
    observation: int
    horizon: int
    step: int = 1


def make_supervised_frame(
    frame: pd.DataFrame,
    *,
    time_col: str,
    target_spec: TargetSpec,
    factor_spec: FactorSpec,
    window: WindowSpec,
    dropna: bool = True,
) -> pd.DataFrame:
    if window.observation < 1:
        raise ValueError("observation window must be at least 1.")
    if window.horizon != target_spec.horizon:
        raise ValueError("window.horizon must match target_spec.horizon.")
    if window.step < 1:
        raise ValueError("window.step must be at least 1.")

    sorted_frame = frame.sort_values(time_col).reset_index(drop=True)
    factors = build_factors(sorted_frame, time_col=time_col, spec=factor_spec)
    target = build_target(sorted_frame, target_spec)

    supervised = pd.concat(
        [sorted_frame[[time_col]].reset_index(drop=True), factors, target.reset_index(drop=True)],
        axis=1,
    )
    if window.step > 1:
        supervised = supervised.iloc[:: window.step].reset_index(drop=True)
    if dropna:
        supervised = supervised.dropna().reset_index(drop=True)
    return supervised

