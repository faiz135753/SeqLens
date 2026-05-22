from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: str
    time_col: str
    target_col: str
    prediction_horizon: int = 1
    sequence_length: int = 30
    moving_average_window: int = 3
    validation_size: float = 0.2
    test_size: float = 0.2
    seed: int = 42
    models: list[str] = field(default_factory=lambda: ["naive", "lstm"])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.__dict__, file, sort_keys=False)
