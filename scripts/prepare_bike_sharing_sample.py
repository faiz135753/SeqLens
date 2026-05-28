from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
    "Bike-Sharing-Dataset.zip"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the UCI Bike Sharing hourly demand benchmark."
    )
    parser.add_argument("--output", default="data/bike_sharing_hourly.csv")
    parser.add_argument("--raw-dir", default="data/raw/bike_sharing")
    args = parser.parse_args()

    output = Path(args.output)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    hour_path = raw_dir / "hour.csv"
    archive_path = raw_dir / "Bike-Sharing-Dataset.zip"
    if not hour_path.exists() and not archive_path.exists():
        archive_path.write_bytes(urlopen(DATASET_URL, timeout=45).read())

    if not hour_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(raw_dir)

    if not hour_path.exists():
        raise SystemExit(f"Missing hourly file after extraction: {hour_path}")

    raw = pd.read_csv(hour_path)
    raw["datetime"] = pd.to_datetime(raw["dteday"]) + pd.to_timedelta(raw["hr"], unit="h")
    frame = raw[
        [
            "datetime",
            "cnt",
            "casual",
            "registered",
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "atemp",
            "hum",
            "windspeed",
        ]
    ].rename(columns={"cnt": "demand"})

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"Wrote {len(frame)} rows to {output}")
    print(frame["demand"].describe().to_string())
    print("Demand quantiles:")
    print(frame["demand"].quantile([0.5, 0.7, 0.8, 0.85, 0.9, 0.95]).to_string())


if __name__ == "__main__":
    main()
