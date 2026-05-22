from __future__ import annotations

import argparse

from seqlens import SeqLens
from seqlens.experiments import ExperimentConfig, run_baseline, run_baseline_comparison
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

    args = parser.parse_args()

    if args.command == "diagnose":
        project = SeqLens(args.csv, time_col=args.time, target_col=args.target)
        diagnostics = project.diagnose()
        suitability = project.score_lstm_suitability()
        print(diagnostics.summary())
        print()
        print(suitability.summary())
    elif args.command == "run-baseline":
        config = ExperimentConfig.from_yaml(args.config)
        result = run_baseline(config, model_name=args.model, output_dir=args.output_dir)
        print(result.summary())
    elif args.command == "compare-baselines":
        config = ExperimentConfig.from_yaml(args.config)
        result = run_baseline_comparison(config, output_dir=args.output_dir)
        print(result.summary())
