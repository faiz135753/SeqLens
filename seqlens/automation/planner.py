from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PromotionDecision:
    stage: str
    status: str
    reason: str


@dataclass(frozen=True)
class ExperimentDiagnosis:
    decisions: list[PromotionDecision]
    next_actions: list[str]

    @property
    def can_promote_to_lgbm(self) -> bool:
        return _decision_status(self.decisions, "lgbm") == "promote"

    @property
    def can_promote_to_lstm(self) -> bool:
        return _decision_status(self.decisions, "lstm") == "promote"


def diagnose_auto_experiment(
    leaderboard: pd.DataFrame,
    distribution: pd.DataFrame,
    extreme_profile: pd.DataFrame | None = None,
    *,
    min_validation_positive_support: int = 30,
    min_rule_test_recall: float = 0.05,
    min_lgbm_test_recall: float = 0.10,
    max_lgbm_false_alarm_rate: float = 0.20,
    max_generalization_gap: float = 0.30,
) -> ExperimentDiagnosis:
    if leaderboard.empty:
        return ExperimentDiagnosis(
            decisions=[
                PromotionDecision(
                    stage="lgbm",
                    status="blocked",
                    reason="No successful candidates were produced.",
                ),
                PromotionDecision(
                    stage="lstm",
                    status="blocked",
                    reason="No successful candidates were produced.",
                ),
            ],
            next_actions=["Inspect errors.csv and fix failed experiment candidates."],
        )

    decisions = []
    next_actions = []
    min_positive_support = int(leaderboard["validation_positive_support"].min())
    if min_positive_support < min_validation_positive_support:
        next_actions.append(
            "Increase data coverage, lower event thresholds, or merge sparse entities."
        )

    rule_models = leaderboard[leaderboard["model"].isin(["recent_window_threshold"])]
    best_rule_recall = _max_metric(rule_models, "test_recall")
    if min_positive_support < min_validation_positive_support:
        decisions.append(
            PromotionDecision(
                stage="lgbm",
                status="caution",
                reason=(
                    "Validation positive support is low; LGBM can be tested, "
                    "but results may be unstable."
                ),
            )
        )
    elif best_rule_recall >= min_rule_test_recall:
        decisions.append(
            PromotionDecision(
                stage="lgbm",
                status="promote",
                reason="A generic event-aware rule found non-zero test recall.",
            )
        )
    else:
        decisions.append(
            PromotionDecision(
                stage="lgbm",
                status="caution",
                reason=(
                    "Simple event rules did not find enough test recall; "
                    "try stronger factors before relying on learned models."
                ),
            )
        )
        next_actions.append("Add rolling, difference, ratio, or domain-specific external factors.")

    lgbm = leaderboard[leaderboard["model"] == "lgbm"]
    if lgbm.empty:
        decisions.append(
            PromotionDecision(
                stage="lstm",
                status="blocked",
                reason="No LGBM candidates were run; use LGBM before LSTM.",
            )
        )
        next_actions.append("Run LGBM candidates after baseline checks.")
    else:
        best_lgbm = lgbm.sort_values(
            ["test_recall", "test_f1"],
            ascending=False,
        ).iloc[0]
        generalization_gap = float(best_lgbm["validation_recall"] - best_lgbm["test_recall"])
        if float(best_lgbm["test_recall"]) < min_lgbm_test_recall:
            decisions.append(
                PromotionDecision(
                    stage="lstm",
                    status="blocked",
                    reason="Best LGBM test recall is too low for LSTM promotion.",
                )
            )
            next_actions.append("Improve factors or event definition before adding sequence models.")
        elif float(best_lgbm["test_false_alarm_rate"]) > max_lgbm_false_alarm_rate:
            decisions.append(
                PromotionDecision(
                    stage="lstm",
                    status="blocked",
                    reason="Best LGBM false alarm rate is too high for LSTM promotion.",
                )
            )
            next_actions.append("Tune threshold strategy against false alarm constraints.")
        elif generalization_gap > max_generalization_gap:
            decisions.append(
                PromotionDecision(
                    stage="lstm",
                    status="blocked",
                    reason="LGBM validation-to-test recall gap is too large.",
                )
            )
            next_actions.append("Use entity-aware or rolling-time validation before LSTM.")
        elif min_positive_support < min_validation_positive_support:
            decisions.append(
                PromotionDecision(
                    stage="lstm",
                    status="blocked",
                    reason="Positive support is too sparse for sequence-model promotion.",
                )
            )
        else:
            decisions.append(
                PromotionDecision(
                    stage="lstm",
                    status="promote",
                    reason="LGBM has usable test recall and acceptable false alarm rate.",
                )
            )

    if _has_sparse_entity_distribution(distribution):
        next_actions.append("Review entity-level event distribution before trusting aggregate metrics.")

    if extreme_profile is not None and not extreme_profile.empty:
        blocked_extremes = extreme_profile[extreme_profile["promotion_gate"] == "blocked"]
        caution_extremes = extreme_profile[extreme_profile["promotion_gate"] == "caution"]
        if not blocked_extremes.empty:
            next_actions.append(
                "Keep blocked extreme-event definitions at the support-planning stage."
            )
        if not caution_extremes.empty:
            next_actions.append(
                "Rank rare-event candidates by recall, miss rate, and false alarm constraints."
            )

    return ExperimentDiagnosis(
        decisions=decisions,
        next_actions=_unique(next_actions),
    )


def diagnosis_to_frame(diagnosis: ExperimentDiagnosis) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": decision.stage,
                "status": decision.status,
                "reason": decision.reason,
            }
            for decision in diagnosis.decisions
        ]
    )


def diagnosis_to_markdown(diagnosis: ExperimentDiagnosis) -> str:
    decisions = diagnosis_to_frame(diagnosis)
    decision_table = (
        "No promotion decisions."
        if decisions.empty
        else decisions.to_markdown(index=False)
    )
    action_lines = "\n".join(f"- {action}" for action in diagnosis.next_actions)
    if not action_lines:
        action_lines = "- Continue with the highest-ranked stable candidate."
    return f"""# Experiment Diagnosis

## Promotion Rules

{decision_table}

## Next Actions

{action_lines}
"""


def _decision_status(decisions: list[PromotionDecision], stage: str) -> str:
    for decision in decisions:
        if decision.stage == stage:
            return decision.status
    return "blocked"


def _max_metric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(frame[column].max())


def _has_sparse_entity_distribution(distribution: pd.DataFrame) -> bool:
    if distribution.empty or "positive_support" not in distribution:
        return False
    return bool((distribution["positive_support"] < 5).any())


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
