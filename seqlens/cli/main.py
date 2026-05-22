from __future__ import annotations

import argparse

from seqlens import SeqLens


def main() -> None:
    parser = argparse.ArgumentParser(prog="seqlens", description="SeqLens research CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    diagnose = subcommands.add_parser("diagnose", help="Diagnose a time-series CSV file")
    diagnose.add_argument("csv", help="Path to the CSV file")
    diagnose.add_argument("--time", required=True, help="Timestamp column name")
    diagnose.add_argument("--target", required=True, help="Target column name")

    args = parser.parse_args()

    if args.command == "diagnose":
        project = SeqLens(args.csv, time_col=args.time, target_col=args.target)
        diagnostics = project.diagnose()
        suitability = project.score_lstm_suitability()
        print(diagnostics.summary())
        print()
        print(suitability.summary())

