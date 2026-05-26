from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExtremeEventPolicy:
    enabled: bool = True
    min_split_positive_support: int = 30
    min_entity_positive_support: int = 5
    rare_event_rate: float = 0.01
    extreme_event_rate: float = 0.001

    @classmethod
    def from_automation(cls, automation: dict) -> "ExtremeEventPolicy":
        raw = automation.get("extreme_events", automation.get("extreme_handling", {}))
        if raw is None:
            raw = {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            min_split_positive_support=int(raw.get("min_split_positive_support", 30)),
            min_entity_positive_support=int(raw.get("min_entity_positive_support", 5)),
            rare_event_rate=float(raw.get("rare_event_rate", 0.01)),
            extreme_event_rate=float(raw.get("extreme_event_rate", 0.001)),
        )


def assess_extreme_event_layer(
    distribution: pd.DataFrame,
    *,
    policy: ExtremeEventPolicy | None = None,
) -> pd.DataFrame:
    policy = policy or ExtremeEventPolicy()
    if not policy.enabled or distribution.empty:
        return pd.DataFrame()

    rows = []
    group_columns = ["threshold", "observation"]
    for (threshold, observation), group in distribution.groupby(group_columns, sort=True):
        split_summary = (
            group.groupby("split", as_index=False)
            .agg({"support": "sum", "positive_support": "sum"})
            .assign(event_rate=lambda frame: frame["positive_support"] / frame["support"])
        )
        support = int(split_summary["support"].sum())
        positive_support = int(split_summary["positive_support"].sum())
        event_rate = 0.0 if support == 0 else positive_support / support
        split_min_positive = int(split_summary["positive_support"].min())
        entity_min_positive = int(group["positive_support"].min())
        imbalance_ratio = _imbalance_ratio(support, positive_support)
        rarity_level = _rarity_level(
            event_rate,
            split_min_positive=split_min_positive,
            entity_min_positive=entity_min_positive,
            policy=policy,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "observation": int(observation),
                "support": support,
                "positive_support": positive_support,
                "event_rate": event_rate,
                "imbalance_ratio": imbalance_ratio,
                "min_split_positive_support": split_min_positive,
                "min_entity_positive_support": entity_min_positive,
                "rarity_level": rarity_level,
                "promotion_gate": _promotion_gate(rarity_level),
                "recommended_action": _recommended_action(rarity_level),
            }
        )
    return pd.DataFrame(rows)


def extreme_event_markdown(profile: pd.DataFrame) -> str:
    if profile.empty:
        table = "Extreme-event handling is disabled or no event distribution rows were available."
    else:
        columns = [
            "threshold",
            "observation",
            "positive_support",
            "event_rate",
            "imbalance_ratio",
            "min_split_positive_support",
            "min_entity_positive_support",
            "rarity_level",
            "promotion_gate",
            "recommended_action",
        ]
        table = profile[columns].to_markdown(index=False, floatfmt=".4f")
    return f"""# Extreme Event Profile

This layer evaluates whether an auto-experiment is operating in a rare or
extreme-value regime before promoting candidates to heavier learned models.

{table}
"""


def _imbalance_ratio(support: int, positive_support: int) -> float | None:
    if positive_support == 0:
        return None
    return float((support - positive_support) / positive_support)


def _rarity_level(
    event_rate: float,
    *,
    split_min_positive: int,
    entity_min_positive: int,
    policy: ExtremeEventPolicy,
) -> str:
    if split_min_positive == 0:
        return "no_validation_signal"
    if (
        event_rate < policy.extreme_event_rate
        or split_min_positive < 10
        or entity_min_positive < policy.min_entity_positive_support
    ):
        return "extreme_sparse"
    if event_rate < policy.rare_event_rate or split_min_positive < policy.min_split_positive_support:
        return "rare"
    if split_min_positive >= 100:
        return "stable"
    return "usable"


def _promotion_gate(rarity_level: str) -> str:
    if rarity_level == "stable":
        return "promote"
    if rarity_level == "usable":
        return "promote_with_monitoring"
    if rarity_level == "rare":
        return "caution"
    return "blocked"


def _recommended_action(rarity_level: str) -> str:
    if rarity_level == "stable":
        return "Use normal model comparison and monitor false alarms."
    if rarity_level == "usable":
        return "Run learned models, but rank by recall, miss rate, and false alarms."
    if rarity_level == "rare":
        return "Prefer recall-aware thresholding and compare event-aware baselines first."
    if rarity_level == "no_validation_signal":
        return "Do not promote; widen the event definition or add more data before modeling."
    return "Treat as an extreme-value task; add data, merge entities, or lower thresholds."
