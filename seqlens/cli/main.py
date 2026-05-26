from __future__ import annotations

import argparse

from seqlens import SeqLens
from seqlens.experiments import (
    EventExperimentConfig,
    ExperimentConfig,
    run_baseline,
    run_baseline_comparison,
    run_event_baseline,
    with_event_model,
)
from seqlens.automation import run_auto_event_experiment
from seqlens.models import SUPPORTED_BASELINES


def main() -> None:
    parser = argparse.ArgumentParser(prog="seqlens", description="SeqLens research CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    diagnose = subcommands.add_parser("diagnose", help="Diagnose a time-series CSV file")
    diagnose.add_argument("csv", help="Path to the CSV file")
    diagnose.add_argument("--time", required=True, help="Timestamp column name")
    diagnose.add_argument("--target", required=True, help="Target column name")

    run_baseline_parser = subcommands.add_parser("run-baseline", help="Run a baseline model")
    run_baseline_parser.add_argument("config", help="Path to the experiment YAML config")
    run_baseline_parser.add_argument(
        "--model",
        default="naive",
        choices=SUPPORTED_BASELINES,
        help="Baseline model to run",
    )
    run_baseline_parser.add_argument(
        "--output-dir",
        default="runs",
        help="Directory for run artifacts",
    )

    compare_baselines_parser = subcommands.add_parser(
        "compare-baselines",
        help="Run all supported baseline models listed in the config",
    )
    compare_baselines_parser.add_argument("config", help="Path to the experiment YAML config")
    compare_baselines_parser.add_argument(
        "--output-dir",
        default="runs",
        help="Directory for comparison artifacts",
    )

    run_event_parser = subcommands.add_parser("run-event", help="Run an event baseline experiment")
    run_event_parser.add_argument("config", help="Path to the event experiment YAML config")
    run_event_parser.add_argument(
        "--model",
        default=None,
        choices=["event_majority", "event_naive", "lgbm"],
        help="Override the event model from config",
    )
    run_event_parser.add_argument("--output-dir", default="runs", help="Directory for artifacts")

    auto_run_parser = subcommands.add_parser("auto-run", help="Run an automated event experiment")
    auto_run_parser.add_argument("config", help="Path to the automated experiment YAML config")
    auto_run_parser.add_argument("--output-dir", default="runs", help="Directory for artifacts")

    args = parser.parse_args()

    if args.command == "diagnose":
        project = SeqLens(args.csv, time_col=args.time, target_col=args.target)
        diagnostics = project.diagnose()
        print(diagnostics.summary())
    elif args.command == "run-baseline":
        config = ExperimentConfig.from_yaml(args.config)
        result = run_baseline(config, model_name=args.model, output_dir=args.output_dir)
        print(result.summary())
    elif args.command == "compare-baselines":
        config = ExperimentConfig.from_yaml(args.config)
        result = run_baseline_comparison(config, output_dir=args.output_dir)
        print(result.summary())
    elif args.command == "run-event":
        config = EventExperimentConfig.from_yaml(args.config)
        if args.model:
            config = with_event_model(config, args.model)
        result = run_event_baseline(config, output_dir=args.output_dir)
        print(result.summary())
    elif args.command == "auto-run":
        result = run_auto_event_experiment(args.config, output_dir=args.output_dir)
        print(result.summary())
