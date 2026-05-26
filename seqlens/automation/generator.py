from __future__ import annotations

from dataclasses import dataclass

from seqlens.factors import FactorSpec
from seqlens.presets import DomainPreset
from seqlens.targets import TargetSpec
from seqlens.windows import WindowSpec


@dataclass(frozen=True)
class ExperimentCandidate:
    name: str
    task_type: str
    target: TargetSpec
    factors: FactorSpec
    window: WindowSpec
    model: str
    primary_metric: str


def generate_candidates(
    preset: DomainPreset,
    *,
    models: list[str],
) -> list[ExperimentCandidate]:
    candidates: list[ExperimentCandidate] = []
    for window in preset.windows:
        for model in models:
            candidates.append(
                ExperimentCandidate(
                    name=f"{preset.name}_{model}_obs{window.observation}_h{window.horizon}",
                    task_type=preset.task_type,
                    target=preset.target,
                    factors=preset.factors,
                    window=window,
                    model=model,
                    primary_metric=preset.primary_metric,
                )
            )
    return candidates

