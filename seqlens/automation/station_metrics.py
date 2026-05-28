from __future__ import annotations

from pathlib import Path

import pandas as pd

from seqlens.evaluation import classification_metrics


def station_level_metrics(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if leaderboard.empty:
        return pd.DataFrame()

    for candidate in leaderboard.itertuples(index=False):
        predictions_path = Path(candidate.run_dir) / "test_predictions.csv"
        if not predictions_path.exists():
            continue
        predictions = pd.read_csv(predictions_path)
        if "entity" not in predictions.columns:
            continue
        for entity, group in predictions.groupby("entity", sort=False):
            metrics = classification_metrics(group["actual"], group["predicted"])
            rows.append(
                {
                    "candidate": candidate.candidate,
                    "threshold": candidate.threshold,
                    "observation": candidate.observation,
                    "model": candidate.model,
                    "threshold_strategy": candidate.threshold_strategy,
                    "entity": entity,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "false_alarm_rate": metrics.false_alarm_rate,
                    "miss_rate": metrics.miss_rate,
                    "event_rate": metrics.event_rate,
                    "support": metrics.support,
                    "positive_support": metrics.positive_support,
                }
            )
    return pd.DataFrame(rows)


def station_level_metrics_markdown(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        table = "No station-level metrics are available."
    else:
        summary = (
            metrics.sort_values(
                ["model", "threshold", "entity", "f1"],
                ascending=[True, True, True, False],
            )
            .groupby(["model", "threshold", "entity"], as_index=False)
            .head(1)
        )
        table = summary.to_markdown(index=False, floatfmt=".4f")
    return f"""# Station-Level Metrics

These metrics show whether aggregate performance is stable across entities.

{table}
"""
