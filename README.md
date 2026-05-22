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
- CLI entry point for quick diagnosis and naive baseline experiments
- MAE, RMSE, MAPE, and direction accuracy metrics

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
from seqlens.experiments import ExperimentConfig, run_naive_baseline

config = ExperimentConfig.from_yaml("configs/basic_lstm.yaml")
result = run_naive_baseline(config)

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
