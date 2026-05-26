# SeqLens

SeqLens is a cross-domain automated time-series experiment framework.

It helps researchers turn raw sequential data into reproducible prediction experiments by defining targets, slicing observation/forecast windows, building factor features, running baseline models, and preparing the path toward LGBM, LSTM, and planner-driven automation.

The project is no longer framed as a single LSTM tool. LSTM and LGBM are model plugins inside a broader experiment system.

## Core Idea

Most forecasting problems can be expressed as:

```text
past observation window -> future prediction window
```

Examples:

- Rainfall: use the past 24 hours to predict whether the next 3 hours exceed 80 mm.
- Stocks: use the past 30 days to predict whether next-day return is positive.
- Energy: use the past 168 hours to predict next-day load.

SeqLens standardizes this process with:

- Target Builder
- Window Slicer
- Factor Builder
- Experiment Generator
- Experiment Engine
- Evaluator
- Reporter
- Planner, planned

## Current MVP

- CSV time-series loading
- Time column and target column validation
- Time-series diagnostics
- Target builders:
  - future value
  - future return
  - future direction
  - future window event
- Factor builders:
  - lag factors
  - rolling factors
  - difference factors
  - ratio-to-rolling-mean factors
  - calendar factors
- Window slicer for supervised experiment frames
- Domain presets:
  - rainfall extreme rain warning
  - stock direction prediction
- Experiment candidate generator
- Naive and moving average baselines
- Baseline comparison reports
- Markdown reports and PNG plots

## Planned Models

SeqLens will run models in increasing complexity:

```text
simple baseline -> LGBM -> LSTM
```

Recommended roles:

- `naive` and `moving_average`: minimum benchmark
- `lgbm`: strong tabular baseline and factor signal detector
- `lstm`: sequence model used only when simpler models show learnable signal
- `llm planner`: future layer that proposes validated experiment configs

## Install

```bash
pip install -e .
```

On macOS, if `python` is unavailable:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Quick Start

Diagnose a CSV:

```bash
seqlens diagnose examples/sample_stock.csv --time date --target close
```

Run a baseline:

```bash
seqlens run-baseline configs/generic_baseline.yaml --model moving_average
```

Compare baselines:

```bash
seqlens compare-baselines configs/generic_baseline.yaml
```

Run an event baseline experiment:

```bash
python scripts/prepare_rainfall_sample.py
seqlens run-event configs/rainfall_extreme_rain.yaml
```

Run an LGBM event model:

```bash
pip install -e ".[lgbm]"
seqlens run-event configs/rainfall_extreme_rain.yaml --model lgbm
```

The first external rainfall test notes are in:

```text
docs/rainfall_event_testing.md
```

## Factor Builder Example

```python
import pandas as pd

from seqlens.factors import FactorSpec, build_factors

frame = pd.read_csv("examples/sample_stock.csv")
spec = FactorSpec(
    lag_columns=["close"],
    lag_periods=[1, 2, 5],
    rolling_columns=["close"],
    rolling_windows=[3, 5],
    rolling_stats=["mean", "std"],
    calendar=["day_of_week", "month"],
)

factors = build_factors(frame, time_col="date", spec=spec)
```

## Rainfall Extreme Rain Concept

```yaml
domain: rainfall
task_type: classification

target:
  type: future_window_event
  column: rainfall
  horizon: 3
  aggregation: sum
  threshold: 80
  name: heavy_rain_next_3h

window:
  observation_windows: [12, 24, 48]
  horizons: [3]

factors:
  lag:
    columns: [rainfall]
    periods: [1, 2, 3, 6, 12, 24]
  rolling:
    columns: [rainfall]
    windows: [3, 6, 12, 24]
    stats: [sum, max, mean, std]
  calendar: [hour, month, season]
```

## Stock Direction Concept

```yaml
domain: stock
task_type: classification

target:
  type: future_direction
  column: close
  horizon: 1

window:
  observation_windows: [10, 20, 30, 60]
  horizons: [1]

factors:
  lag:
    columns: [close, volume]
    periods: [1, 2, 5, 10, 20]
  rolling:
    columns: [close]
    windows: [5, 10, 20]
    stats: [mean, std, min, max]
  transforms:
    - return
    - ratio_to_rolling_mean
```

## Repository Layout

```text
seqlens/
├── automation/      # experiment candidate generation and future planners
├── data/            # data loading and time-based splitting
├── diagnostics/     # time-series quality checks
├── factors/         # lag, rolling, ratio, difference, calendar factors
├── models/          # baselines now; LGBM/LSTM plugins later
├── presets/         # rainfall, stock, generic presets
├── reports/         # markdown and plot outputs
├── targets/         # future targets and event labels
├── windows/         # supervised slicing
└── experiments/     # runnable experiment engine
```

## Design Rule

LLM or automated planners should not edit arbitrary training code. They should generate structured experiment candidates that SeqLens validates and executes.
