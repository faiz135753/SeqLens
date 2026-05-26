from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from seqlens.experiments import (
    EventExperimentConfig,
    EventRunResult,
    make_event_supervised_dataset,
    run_event_baseline,
    with_event_model,
    with_event_threshold,
    with_observation_window,
)
from seqlens.data.splitting import time_based_split


@dataclass(frozen=True)
class AutoExperimentResult:
    run_dir: Path
    leaderboard_path: Path
    distribution_path: Path
    recommendations_path: Path
    report_path: Path
    candidate_count: int

    def summary(self) -> str:
        return (
            f"Auto experiment directory: {self.run_dir}\n"
            f"Candidates: {self.candidate_count}\n"
            f"Leaderboard: {self.leaderboard_path}\n"
            f"Event distribution: {self.distribution_path}\n"
            f"Recommendations: {self.recommendations_path}\n"
            f"Final report: {self.report_path}"
        )


def run_auto_event_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path = "runs",
) -> AutoExperimentResult:
    config_path = Path(config_path)
    raw = _load_yaml(config_path)
    base_config = EventExperimentConfig.from_yaml(config_path)
    thresholds = _automation_values(
        raw,
        key="thresholds",
        fallback=[base_config.target.threshold],
    )
    observation_windows = _automation_values(
        raw,
        key="observation_windows",
        fallback=[base_config.window.observation],
    )
    models = _automation_values(
        raw,
        key="models",
        fallback=["event_majority", "lgbm"],
    )

    run_dir = _create_run_dir(output_dir, name="auto_event")
    frame = pd.read_csv(base_config.data_path)
    distribution = _event_distribution_report(
        frame=frame,
        base_config=base_config,
        thresholds=thresholds,
        observation_windows=observation_windows,
    )
    distribution_path = run_dir / "event_distribution.csv"
    distribution.to_csv(distribution_path, index=False)
    distribution_md_path = run_dir / "event_distribution.md"
    distribution_md_path.write_text(_event_distribution_markdown(distribution), encoding="utf-8")

    rows = []
    errors = []
    for threshold in thresholds:
        if threshold is None:
            continue
        for observation in observation_windows:
            for model in models:
                candidate_name = (
                    f"thr{_format_value(threshold)}_obs{int(observation)}_{model}"
                )
                candidate_dir = run_dir / candidate_name
                candidate_config = with_event_model(
                    with_observation_window(
                        with_event_threshold(base_config, float(threshold)),
                        int(observation),
                    ),
                    str(model),
                )
                try:
                    result = run_event_baseline(candidate_config, output_dir=candidate_dir)
                    rows.append(_leaderboard_row(candidate_name, threshold, observation, model, result))
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "candidate": candidate_name,
                            "threshold": threshold,
                            "observation": observation,
                            "model": model,
                            "error": str(exc),
                        }
                    )

    leaderboard = pd.DataFrame(rows)
    leaderboard_path = run_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(run_dir / "errors.csv", index=False)

    recommendations = _recommendations(leaderboard, errors, distribution)
    recommendations_path = run_dir / "recommendations.md"
    recommendations_path.write_text(recommendations, encoding="utf-8")

    report_path = run_dir / "final_report.md"
    report_path.write_text(
        _final_report(
            leaderboard=leaderboard,
            distribution=distribution,
            errors=errors,
            recommendations=recommendations,
        ),
        encoding="utf-8",
    )

    return AutoExperimentResult(
        run_dir=run_dir,
        leaderboard_path=leaderboard_path,
        distribution_path=distribution_path,
        recommendations_path=recommendations_path,
        report_path=report_path,
        candidate_count=len(rows) + len(errors),
    )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _automation_values(raw: dict, *, key: str, fallback: list) -> list:
    automation = raw.get("automation", {})
    value = automation.get(key)
    if value is None:
        return fallback
    return value


def _create_run_dir(output_dir: str | Path, *, name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_dir)
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index}"
        run_dir = base_dir / f"{timestamp}_{name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique auto experiment directory under {base_dir}")


def _leaderboard_row(
    candidate_name: str,
    threshold: float,
    observation: int,
    model: str,
    result: EventRunResult,
) -> dict[str, float | int | str | None]:
    return {
        "candidate": candidate_name,
        "threshold": threshold,
        "observation": observation,
        "model": model,
        "run_dir": str(result.run_dir),
        "decision_threshold": result.threshold,
        "event_rate": result.event_rate,
        "supervised_rows": result.supervised_rows,
        "validation_precision": result.validation_metrics.precision,
        "validation_recall": result.validation_metrics.recall,
        "validation_f1": result.validation_metrics.f1,
        "validation_false_alarm_rate": result.validation_metrics.false_alarm_rate,
        "validation_miss_rate": result.validation_metrics.miss_rate,
        "validation_positive_support": result.validation_metrics.positive_support,
        "test_precision": result.test_metrics.precision,
        "test_recall": result.test_metrics.recall,
        "test_f1": result.test_metrics.f1,
        "test_false_alarm_rate": result.test_metrics.false_alarm_rate,
        "test_miss_rate": result.test_metrics.miss_rate,
        "test_positive_support": result.test_metrics.positive_support,
    }


def _event_distribution_report(
    *,
    frame: pd.DataFrame,
    base_config: EventExperimentConfig,
    thresholds: list,
    observation_windows: list,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        if threshold is None:
            continue
        for observation in observation_windows:
            config = with_observation_window(
                with_event_threshold(base_config, float(threshold)),
                int(observation),
            )
            supervised = make_event_supervised_dataset(frame, config)
            split = time_based_split(
                supervised,
                validation_size=config.validation_size,
                test_size=config.test_size,
            )
            target_col = config.target.output_name
            for split_name, split_frame in [
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            ]:
                rows.extend(
                    _distribution_rows(
                        split_frame,
                        split_name=split_name,
                        threshold=float(threshold),
                        observation=int(observation),
                        target_col=target_col,
                        entity_col=config.entity_col,
                    )
                )
    return pd.DataFrame(rows)


def _distribution_rows(
    frame: pd.DataFrame,
    *,
    split_name: str,
    threshold: float,
    observation: int,
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
                "observation": observation,
                "split": split_name,
                "entity": str(entity),
                "support": support,
                "positive_support": positives,
                "event_rate": 0.0 if support == 0 else positives / support,
            }
        )
    return rows


def _event_distribution_markdown(distribution: pd.DataFrame) -> str:
    if distribution.empty:
        table = "No event distribution rows."
    else:
        summary = (
            distribution.groupby(["threshold", "observation", "split"], as_index=False)
            .agg({"support": "sum", "positive_support": "sum"})
            .assign(event_rate=lambda frame: frame["positive_support"] / frame["support"])
        )
        table = summary.to_markdown(index=False, floatfmt=".4f")
    return f"""# Event Distribution

## Split Summary

{table}
"""


def _recommendations(
    leaderboard: pd.DataFrame,
    errors: list[dict],
    distribution: pd.DataFrame,
) -> str:
    lines = ["# Recommendations", ""]
    if leaderboard.empty:
        lines.append("No candidates completed successfully. Inspect `errors.csv`.")
        return "\n".join(lines)

    best = leaderboard.sort_values(
        ["validation_f1", "validation_recall"],
        ascending=False,
    ).iloc[0]
    lines.extend(
        [
            "## Best Validation Candidate",
            "",
            f"- Candidate: `{best['candidate']}`",
            f"- Model: `{best['model']}`",
            f"- Threshold: `{best['threshold']}`",
            f"- Observation window: `{best['observation']}`",
            f"- Validation recall: `{best['validation_recall']:.3f}`",
            f"- Validation F1: `{best['validation_f1']:.3f}`",
            f"- Test recall: `{best['test_recall']:.3f}`",
            f"- Test F1: `{best['test_f1']:.3f}`",
            "",
        ]
    )

    min_positive_support = int(leaderboard["validation_positive_support"].min())
    if min_positive_support < 30:
        lines.extend(
            [
                "## Data Issue",
                "",
                "- Some candidates have fewer than 30 validation positive events.",
                "- Expand years, add stations, or lower the threshold before promoting to LSTM.",
                "",
            ]
        )

    lgbm = leaderboard[leaderboard["model"] == "lgbm"]
    if not lgbm.empty and float(lgbm["test_recall"].max()) <= 0:
        lines.extend(
            [
                "## Model Issue",
                "",
                "- LGBM failed to recall events on the test split.",
                "- Do not promote this task to LSTM yet.",
                "- Add meteorological factors and station-aware validation first.",
                "",
            ]
        )
    elif not lgbm.empty:
        lines.extend(
            [
                "## Model Signal",
                "",
                "- At least one LGBM candidate found non-zero test recall.",
                "- Compare station-level event distribution before considering LSTM.",
                "",
            ]
        )

    if errors:
        lines.extend(
            [
                "## Execution Issue",
                "",
                f"- `{len(errors)}` candidates failed. Inspect `errors.csv`.",
                "",
            ]
        )

    sparse = distribution[distribution["positive_support"] < 5]
    if not sparse.empty:
        sparse_splits = sparse[["threshold", "observation", "split", "entity"]].head(10)
        lines.extend(
            [
                "## Station Distribution Issue",
                "",
                "- Some station-level splits have fewer than 5 positive events.",
                "- This can make validation or test recall unstable.",
                "",
                sparse_splits.to_markdown(index=False),
                "",
            ]
        )

    lines.extend(
        [
            "## Next Experiment Suggestions",
            "",
            "- Add station-level event distribution artifacts.",
            "- Add threshold strategy modes beyond F1 maximization.",
            "- Add rainfall-aware baselines such as rolling sum threshold.",
            "- Add humidity, pressure, wind, and pressure-change factors when available.",
        ]
    )
    return "\n".join(lines)


def _final_report(
    leaderboard: pd.DataFrame,
    distribution: pd.DataFrame,
    errors: list[dict],
    recommendations: str,
) -> str:
    if leaderboard.empty:
        table = "No successful candidates."
    else:
        table = leaderboard.sort_values(
            ["validation_f1", "validation_recall"],
            ascending=False,
        ).to_markdown(index=False, floatfmt=".3f")
    return f"""# SeqLens Auto Experiment Report

## Leaderboard

{table}

## Event Distribution Summary

{_event_distribution_markdown(distribution)}

## Failed Candidates

{len(errors)}

{recommendations}
"""


def _format_value(value: object) -> str:
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value).replace(".", "p")
