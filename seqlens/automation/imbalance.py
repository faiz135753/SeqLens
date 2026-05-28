from __future__ import annotations

import pandas as pd


def imbalance_diagnosis(support_plan: pd.DataFrame) -> pd.DataFrame:
    if support_plan.empty:
        return pd.DataFrame(
            columns=[
                "threshold",
                "horizon",
                "min_positive_support",
                "mean_event_rate",
                "severity",
                "recommended_strategy",
            ]
        )

    summary = (
        support_plan.groupby(["threshold", "horizon", "split"], as_index=False)
        .agg({"support": "sum", "positive_support": "sum"})
        .assign(event_rate=lambda frame: frame["positive_support"] / frame["support"])
    )
    diagnosis = (
        summary.groupby(["threshold", "horizon"], as_index=False)
        .agg(
            min_positive_support=("positive_support", "min"),
            mean_event_rate=("event_rate", "mean"),
        )
        .sort_values(["threshold", "horizon"])
        .reset_index(drop=True)
    )
    diagnosis["severity"] = diagnosis.apply(
        lambda row: _severity(
            int(row["min_positive_support"]),
            float(row["mean_event_rate"]),
        ),
        axis=1,
    )
    diagnosis["recommended_strategy"] = diagnosis["severity"].map(_strategy)
    return diagnosis


def imbalance_diagnosis_markdown(diagnosis: pd.DataFrame) -> str:
    if diagnosis.empty:
        table = "No imbalance diagnosis rows."
    else:
        table = diagnosis.to_markdown(index=False, floatfmt=".4f")
    return f"""# Imbalance Diagnosis

This artifact classifies each event definition by rarity and recommends the
next automated-experiment strategy.

{table}
"""


def _severity(min_positive_support: int, mean_event_rate: float) -> str:
    if min_positive_support < 10 or mean_event_rate < 0.0005:
        return "unusable"
    if min_positive_support < 30 or mean_event_rate < 0.001:
        return "extreme"
    if min_positive_support < 100 or mean_event_rate < 0.005:
        return "rare"
    if mean_event_rate < 0.05:
        return "imbalanced"
    return "normal"


def _strategy(severity: str) -> str:
    if severity == "unusable":
        return "Do not train learned models; reformulate threshold, horizon, or target."
    if severity == "extreme":
        return "Use support planning, longer horizons, and simple baselines before LGBM."
    if severity == "rare":
        return "Allow cautious LGBM with class weighting and false-alarm constraints."
    if severity == "imbalanced":
        return "Use imbalance-aware metrics, threshold tuning, and validation stability checks."
    return "Standard automated experimentation is acceptable."
