# SeqLens Design Flow

SeqLens is designed as a guardrailed LSTM research assistant, not a stock prediction promise machine.

The project should help users answer three practical questions:

1. Is my time-series data clean enough for sequence modeling?
2. Is LSTM a reasonable model to try, or are simple baselines enough?
3. If the model performs poorly, what should I try next without breaking research validity?

## System Flow

```text
Load CSV data
→ Validate time and target columns
→ Diagnose time-series quality
→ Score LSTM suitability
→ Create baseline experiments
→ Train candidate models
→ Evaluate with validation metrics
→ Let planner suggest the next experiment config
→ Execute only validated configs
→ Save config, seed, metrics, plots, and reasoning
→ Generate research report
```

## Core Modules

```text
data
  Load and validate user datasets.

diagnostics
  Detect missing values, duplicate timestamps, irregular intervals, outliers,
  autocorrelation, and trend.

suitability
  Convert diagnostics into an LSTM suitability score with reasons and warnings.

experiments
  Store reproducible experiment configs and eventually run folders.

models
  Planned: naive, moving average, ARIMA, tree-based models, LSTM, GRU.

evaluation
  Planned: MAE, RMSE, MAPE, sMAPE, direction accuracy, training time.

agent
  Planned: restricted LLM planner that proposes the next config but never runs
  arbitrary code.
```

## LLM-Guided Optimization

The LLM should act as a research planner, not as an unrestricted programmer.

Allowed:

- Read diagnostics and metrics
- Explain likely failure modes
- Propose the next experiment as structured YAML or JSON
- Recommend whether to stop, continue, or simplify the model

Not allowed:

- Modify training code in normal user mode
- Use test metrics for iterative tuning
- Run unlimited training loops
- Bypass data leakage checks
- Produce investment advice

## Planned Agent Contract

Future versions should expose a small tool surface:

```text
diagnose_data(config)
run_experiment(config)
evaluate_run(run_id)
compare_runs(run_ids)
suggest_next_config(context)
generate_report(run_id)
```

The LLM output should be validated against a schema before execution:

```yaml
sequence_length: 30
hidden_units: 64
dropout: 0.2
learning_rate: 0.001
batch_size: 32
features:
  - close
  - volume
objective: minimize_validation_rmse
reason: Reduce overfitting while preserving recent market signal.
```

## Financial Time-Series Guardrails

For stock-like data, SeqLens should eventually check:

- Train / validation / test split is time-based
- Scalers are fit only on training data
- Rolling indicators do not use future values
- Test data is used only once for final reporting
- Baselines are compared before promoting LSTM results
- Direction accuracy is reported alongside regression metrics
- Reports clearly state that forecasts are research outputs, not trading advice

## MVP Boundary

Current MVP includes:

- Python package structure
- CSV loader
- Time-series diagnostics
- LSTM suitability score
- CLI diagnosis command
- Example config and sample CSV
- Basic test coverage

Next development milestone:

- Add train / validation / test splitting
- Add naive forecast baseline
- Add MAE, RMSE, MAPE metrics
- Save run artifacts under `runs/`

