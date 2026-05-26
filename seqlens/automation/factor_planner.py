from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from seqlens.experiments import EventExperimentConfig


@dataclass(frozen=True)
class FactorRecommendation:
    column: str
    factor_type: str
    windows: str
    parameters: str
    reason: str


def recommend_event_factors(
    frame: pd.DataFrame,
    config: EventExperimentConfig,
) -> list[FactorRecommendation]:
    numeric_columns = _numeric_columns(frame)
    excluded = {config.time_col}
    if config.entity_col:
        excluded.add(config.entity_col)

    target_column = config.target.column
    recommendations = []
    if target_column in numeric_columns:
        recommendations.extend(
            [
                FactorRecommendation(
                    column=target_column,
                    factor_type="rolling",
                    windows="3,6,12,24",
                    parameters="sum,max,std,nonzero_count,slope",
                    reason="Capture recent intensity, activity count, volatility, and trend.",
                ),
                FactorRecommendation(
                    column=target_column,
                    factor_type="diff",
                    windows="1,3,6",
                    parameters="difference",
                    reason="Capture short-term acceleration or regime change.",
                ),
                FactorRecommendation(
                    column=target_column,
                    factor_type="ratio_to_rolling_mean",
                    windows="6,12,24",
                    parameters="current / rolling_mean",
                    reason="Capture abnormal level compared with recent baseline.",
                ),
            ]
        )

    for column in numeric_columns:
        if column in excluded or column == target_column:
            continue
        recommendations.extend(
            [
                FactorRecommendation(
                    column=column,
                    factor_type="rolling",
                    windows="3,6,12,24",
                    parameters="mean,std,slope",
                    reason="Use external numeric signals as contextual predictors.",
                ),
                FactorRecommendation(
                    column=column,
                    factor_type="diff",
                    windows="1,3,6",
                    parameters="difference",
                    reason="Capture recent external-signal changes before an event.",
                ),
            ]
        )

    return recommendations


def factor_recommendations_to_frame(
    recommendations: list[FactorRecommendation],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": recommendation.column,
                "factor_type": recommendation.factor_type,
                "windows": recommendation.windows,
                "parameters": recommendation.parameters,
                "reason": recommendation.reason,
            }
            for recommendation in recommendations
        ]
    )


def factor_recommendations_to_markdown(
    recommendations: list[FactorRecommendation],
) -> str:
    frame = factor_recommendations_to_frame(recommendations)
    table = "No factor recommendations." if frame.empty else frame.to_markdown(index=False)
    return f"""# Factor Recommendations

These recommendations are generic next-round feature recipes for event prediction.

{table}
"""


def _numeric_columns(frame: pd.DataFrame) -> set[str]:
    numeric = set()
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            numeric.add(column)
    return numeric
