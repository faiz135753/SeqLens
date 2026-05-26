from __future__ import annotations

from dataclasses import replace

import pandas as pd

from seqlens.data.splitting import time_based_split
from seqlens.experiments import EventExperimentConfig
from seqlens.targets import build_target


def event_support_plan(
    frame: pd.DataFrame,
    config: EventExperimentConfig,
    *,
    thresholds: list,
    horizons: list,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        if threshold is None:
            continue
        for horizon in horizons:
            target_config = replace(
                config,
                target=replace(
                    config.target,
                    threshold=float(threshold),
                    horizon=int(horizon),
                    name=(
                        f"{config.target.column}_{config.target.aggregation}_next_"
                        f"{int(horizon)}_ge_{_format_value(threshold)}"
                    ),
                ),
                window=replace(config.window, horizon=int(horizon)),
            )
            target_frame = _event_target_frame(frame, target_config)
            split = time_based_split(
                target_frame,
                validation_size=config.validation_size,
                test_size=config.test_size,
            )
            target_col = target_config.target.output_name
            for split_name, split_frame in [
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            ]:
                rows.extend(
                    _support_rows(
                        split_frame,
                        split_name=split_name,
                        threshold=float(threshold),
                        horizon=int(horizon),
                        target_col=target_col,
                        entity_col=config.entity_col,
                    )
                )
    return pd.DataFrame(rows)


def event_support_markdown(support: pd.DataFrame) -> str:
    if support.empty:
        table = "No event support rows."
    else:
        summary = (
            support.groupby(["threshold", "horizon", "split"], as_index=False)
            .agg({"support": "sum", "positive_support": "sum"})
            .assign(event_rate=lambda frame: frame["positive_support"] / frame["support"])
        )
        table = summary.to_markdown(index=False, floatfmt=".4f")
    return f"""# Event Support Plan

Use this artifact to choose event definitions with enough positive samples before
promoting an experiment to learned models.

{table}
"""


def _event_target_frame(frame: pd.DataFrame, config: EventExperimentConfig) -> pd.DataFrame:
    if config.entity_col and config.entity_col in frame.columns:
        pieces = []
        for entity_value, group in frame.groupby(config.entity_col, sort=False):
            target = build_target(group.sort_values(config.time_col), config.target)
            piece = pd.DataFrame(
                {
                    config.time_col: group.sort_values(config.time_col)[config.time_col].to_numpy(),
                    config.entity_col: entity_value,
                    config.target.output_name: target.to_numpy(),
                }
            )
            pieces.append(piece)
        result = pd.concat(pieces, ignore_index=True)
    else:
        sorted_frame = frame.sort_values(config.time_col).reset_index(drop=True)
        target = build_target(sorted_frame, config.target)
        result = pd.DataFrame(
            {
                config.time_col: sorted_frame[config.time_col],
                config.target.output_name: target,
            }
        )
    return result.dropna().sort_values(config.time_col).reset_index(drop=True)


def _support_rows(
    frame: pd.DataFrame,
    *,
    split_name: str,
    threshold: float,
    horizon: int,
    target_col: str,
    entity_col: str | None,
) -> list[dict[str, float | int | str]]:
    if entity_col and entity_col in frame.columns:
        groups = frame.groupby(entity_col, sort=False)
    else:
        groups = [("all", frame)]

    rows = []
    for entity, group in groups:
        support = int(len(group))
        positives = int((group[target_col] == 1).sum())
        rows.append(
            {
                "threshold": threshold,
                "horizon": horizon,
                "split": split_name,
                "entity": str(entity),
                "support": support,
                "positive_support": positives,
                "event_rate": 0.0 if support == 0 else positives / support,
                "support_status": _support_status(positives),
            }
        )
    return rows


def _support_status(positive_support: int) -> str:
    if positive_support >= 100:
        return "stable"
    if positive_support >= 30:
        return "usable"
    if positive_support >= 10:
        return "sparse"
    return "too_sparse"


def _format_value(value: object) -> str:
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value).replace(".", "p")
