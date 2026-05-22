# SeqLens

SeqLens is an open-source research assistant for LSTM-based time-series work.

It helps learners and researchers diagnose time-series data, evaluate whether an LSTM is a reasonable modeling choice, compare baselines, and keep experiments reproducible before adding LLM-guided optimization.

## Project Goal

Many beginners can write an LSTM model but struggle to decide what to change when the model performs poorly. SeqLens is designed to become a guardrailed experiment framework where an LLM can safely plan the next experiment while the library handles data checks, leakage prevention, metrics, and reproducible records.

## MVP Features

- CSV time-series loading
- Time column and target column validation
- Missing value, duplicate timestamp, frequency, and outlier diagnostics
- Basic trend and autocorrelation diagnostics
- LSTM suitability scoring with human-readable reasons
- YAML experiment configuration
- CLI entry point for quick diagnosis and baseline experiments
- Naive and moving average baselines
- MAE, RMSE, MAPE, and direction accuracy metrics
- Markdown report and actual-vs-predicted PNG plots

## Planned Features

- Naive, moving average, ARIMA, tree-based, LSTM, and GRU baselines
- Train / validation / test split guardrails
- Feature engineering for lag, rolling, and financial indicators
- Experiment folders with config, metrics, plots, and reports
- LLM-guided experiment planning through restricted configuration tools
- Streamlit interface for beginner-friendly analysis

## Install

```bash
pip install -e .
```

On macOS, if `python` is unavailable, use:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

For optional statistical diagnostics:

```bash
pip install -e ".[diagnostics]"
```

## Quick Start

```bash
seqlens diagnose examples/sample_stock.csv --time date --target close
```

Run the first baseline experiment:

```bash
seqlens run-baseline configs/basic_lstm.yaml
```

Run a moving average baseline:

```bash
seqlens run-baseline configs/basic_lstm.yaml --model moving_average
```

Compare all supported baselines listed in the config:

```bash
seqlens compare-baselines configs/basic_lstm.yaml
```

Each run writes artifacts under `runs/`:

```text
runs/<timestamp>_<model>/
├── config.yaml
├── metadata.yaml
├── metrics.json
├── validation_predictions.csv
├── test_predictions.csv
├── validation_actual_vs_predicted.png
├── test_actual_vs_predicted.png
└── report.md
```

Baseline comparison runs also write:

```text
runs/<timestamp>_baseline_comparison/
├── comparison.csv
├── comparison_report.md
├── <timestamp>_naive/
└── <timestamp>_moving_average/
```

Python API:

```python
from seqlens import SeqLens

project = SeqLens("examples/sample_stock.csv", time_col="date", target_col="close")
diagnostics = project.diagnose()
score = project.score_lstm_suitability()

print(diagnostics.summary())
print(score.summary())
```

Baseline experiment:

```python
from seqlens.experiments import ExperimentConfig, run_baseline

config = ExperimentConfig.from_yaml("configs/basic_lstm.yaml")
result = run_baseline(config, model_name="moving_average")

print(result.summary())
```

## Design Principle

SeqLens should not promise that LSTM can predict stocks reliably. Instead, it helps users run cleaner experiments:

- use time-based splits
- avoid look-ahead bias
- compare against simple baselines
- record every experiment
- explain why a model may be failing

## Repository Layout

```text
seqlens/
├── seqlens/
│   ├── data/
│   ├── diagnostics/
│   ├── evaluation/
│   ├── models/
│   ├── suitability/
│   ├── experiments/
│   └── cli/
├── configs/
├── examples/
└── tests/
```
