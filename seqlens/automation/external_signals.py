from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExternalSignalRecommendation:
    trigger_reason: str
    suggested_signal_group: str
    examples: str
    expected_value: str
    priority: int


def recommend_external_signals(
    leaderboard: pd.DataFrame,
    imbalance: pd.DataFrame,
    *,
    domain: str = "generic",
    false_alarm_cap: float = 0.05,
) -> list[ExternalSignalRecommendation]:
    if leaderboard.empty:
        return []

    trigger_reasons = _trigger_reasons(
        leaderboard,
        imbalance,
        false_alarm_cap=false_alarm_cap,
    )
    if not trigger_reasons:
        return []

    return [
        ExternalSignalRecommendation(
            trigger_reason="; ".join(trigger_reasons),
            suggested_signal_group=group,
            examples=examples,
            expected_value=value,
            priority=priority,
        )
        for group, examples, value, priority in _signal_groups(domain)
    ]


def external_signal_recommendations_to_frame(
    recommendations: list[ExternalSignalRecommendation],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trigger_reason": recommendation.trigger_reason,
                "suggested_signal_group": recommendation.suggested_signal_group,
                "examples": recommendation.examples,
                "expected_value": recommendation.expected_value,
                "priority": recommendation.priority,
            }
            for recommendation in recommendations
        ]
    )


def external_signal_recommendations_to_markdown(
    recommendations: list[ExternalSignalRecommendation],
) -> str:
    frame = external_signal_recommendations_to_frame(recommendations)
    table = (
        "External signals were not triggered by the current experiment."
        if frame.empty
        else frame.sort_values(["priority", "suggested_signal_group"]).to_markdown(index=False)
    )
    return f"""# External Signal Recommendations

This artifact is triggered when internal time-series factors appear insufficient
or learned models cannot improve reliably over simple event-aware baselines.

{table}
"""


def _trigger_reasons(
    leaderboard: pd.DataFrame,
    imbalance: pd.DataFrame,
    *,
    false_alarm_cap: float,
) -> list[str]:
    reasons = []
    if _has_unusable_or_extreme_definitions(imbalance):
        reasons.append("Some event definitions are too sparse for learned models.")

    baseline = leaderboard[leaderboard["model"] == "recent_window_threshold"]
    lgbm = leaderboard[leaderboard["model"] == "lgbm"]
    if lgbm.empty:
        return reasons

    best_lgbm = lgbm.sort_values(["test_f1", "test_recall"], ascending=False).iloc[0]
    if not baseline.empty:
        best_baseline = baseline.sort_values(
            ["test_f1", "test_recall"],
            ascending=False,
        ).iloc[0]
        if float(best_baseline["test_f1"]) >= float(best_lgbm["test_f1"]):
            reasons.append("A simple event-aware baseline outperforms the best LGBM F1.")
        elif float(best_lgbm["test_f1"]) < float(best_baseline["test_f1"]) * 1.1:
            reasons.append("LGBM does not clearly improve over the event-aware baseline.")

    max_lgbm_false_alarm = float(lgbm["test_false_alarm_rate"].max())
    if max_lgbm_false_alarm > false_alarm_cap:
        reasons.append("LGBM requires a high false-alarm rate to recover events.")

    high_recall = float(lgbm["test_recall"].max()) >= 0.2
    weak_f1 = float(best_lgbm["test_f1"]) < 0.1
    if high_recall and weak_f1:
        reasons.append("LGBM finds recall but has weak precision/F1 stability.")

    return _unique(reasons)


def _has_unusable_or_extreme_definitions(imbalance: pd.DataFrame) -> bool:
    if imbalance.empty or "severity" not in imbalance:
        return False
    return bool(imbalance["severity"].isin(["unusable", "extreme"]).any())


def _signal_groups(domain: str) -> list[tuple[str, str, str, int]]:
    generic = [
        (
            "context_variables",
            "environment, market, demand, load, operational state",
            "Add drivers that explain why the target enters an extreme regime.",
            1,
        ),
        (
            "temporal_change",
            "pressure_diff, humidity_diff, volatility_change, load_ramp",
            "Capture precursor changes before the extreme event.",
            1,
        ),
        (
            "spatial_or_peer_context",
            "nearby stations, peer assets, upstream sensors, sector index",
            "Recover signals that are visible outside the target entity.",
            2,
        ),
        (
            "entity_metadata",
            "latitude, longitude, elevation, asset type, industry, capacity",
            "Help models separate entity-level heterogeneity.",
            2,
        ),
        (
            "regime_indicators",
            "season, typhoon/front labels, macro regime, maintenance mode",
            "Represent external regimes that change event probability.",
            3,
        ),
    ]
    if domain == "rainfall":
        return [
            (
                "same_station_weather",
                "pressure, humidity, temperature, wind_speed, wind_direction",
                "Add atmospheric state beyond rainfall-only history.",
                1,
            ),
            (
                "weather_change_rates",
                "pressure_drop, humidity_rise, wind_shift, temperature_change",
                "Capture physical precursors of heavy rainfall.",
                1,
            ),
            (
                "spatial_rainfall_context",
                "nearby_station_rainfall, regional_max_rainfall, upstream_rainfall",
                "Heavy rainfall often has spatial structure not visible at one station.",
                2,
            ),
            (
                "station_metadata",
                "latitude, longitude, elevation, region, terrain class",
                "Model station heterogeneity and orographic effects.",
                2,
            ),
            (
                "weather_regime",
                "typhoon distance, front label, radar echo, satellite cloud features",
                "Represent large-scale systems that drive extreme rainfall.",
                3,
            ),
        ]
    return generic


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
