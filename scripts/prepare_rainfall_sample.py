from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


DEFAULT_STATIONS = {
    "466920": "Taipei",
    "466940": "Keelung",
    "467490": "Taichung",
    "467530": "Alishan",
    "466990": "Hualien",
    "467660": "Taitung",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a multi-station hourly rainfall sample.")
    parser.add_argument("--output", default="data/rainfall_hourly.csv")
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023])
    parser.add_argument("--raw-dir", default="data/raw/rainfall")
    args = parser.parse_args()

    output = Path(args.output)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    skipped = []
    for station_id, station_name in DEFAULT_STATIONS.items():
        for year in args.years:
            url = (
                "https://raw.githubusercontent.com/Raingel/historical_weather/main/"
                f"data/{station_id}/{station_id}_{year}.csv"
            )
            raw_path = raw_dir / f"{station_id}_{year}.csv"
            try:
                raw_path.write_bytes(urlopen(url, timeout=45).read())
                raw = pd.read_csv(raw_path, encoding="utf-8-sig")
            except Exception as exc:  # noqa: BLE001
                skipped.append({"station_id": station_id, "year": year, "reason": str(exc)})
                continue

            time_col = raw.columns[0]
            if "Precp" not in raw.columns:
                skipped.append({"station_id": station_id, "year": year, "reason": "missing Precp"})
                continue

            frame = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(raw[time_col], errors="coerce"),
                    "station_id": station_id,
                    "station_name": station_name,
                    "rainfall": pd.to_numeric(raw["Precp"], errors="coerce"),
                }
            )
            frame["rainfall"] = frame["rainfall"].clip(lower=0).fillna(0)
            frame = frame.dropna(subset=["datetime"])
            frames.append(frame)

    if not frames:
        raise SystemExit("No rainfall files could be downloaded.")

    combined = pd.concat(frames, ignore_index=True).sort_values(["datetime", "station_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)

    print(f"Wrote {len(combined)} rows to {output}")
    print(combined.groupby("station_id")["rainfall"].agg(["count", "sum", "max"]).to_string())
    if skipped:
        print("Skipped files:")
        for item in skipped:
            print(f"- {item['station_id']} {item['year']}: {item['reason']}")


if __name__ == "__main__":
    main()

