# SeqLens Redesign

SeqLens is being rebuilt as a cross-domain automated time-series experiment framework.

The central abstraction is:

```text
past observation window -> future prediction window
```

Any domain-specific task should be decomposed into:

1. Data Adapter
2. Domain Preset
3. Target Builder
4. Window Slicer
5. Factor Builder
6. Experiment Generator
7. Experiment Engine
8. Evaluator
9. Reporter
10. Planner

## Why This Design

Rainfall, stock, energy, sales, and sensor problems look different, but the experiment structure is similar:

```text
define target
slice history and future windows
generate factors
run cheap baselines
run LGBM as a strong factor model
promote to LSTM only if sequence modeling is justified
compare models
write reports
```

This makes LSTM and LGBM model plugins, not the core architecture.

## Target Builder

Turns raw columns into supervised labels.

Supported MVP target types:

```text
future_value
future_return
future_direction
future_window_event
```

Examples:

```text
Stock:
close_direction_next_1 = close[t+1] > close[t]

Rainfall:
heavy_rain_next_3h = sum(rainfall[t+1:t+3]) >= 80
```

## Window Slicer

Defines how observations become model-ready samples.

```yaml
window:
  observation_windows: [12, 24, 48]
  horizons: [1, 3, 6]
  step: 1
```

For LGBM, the slicer produces a 2D tabular factor matrix.

```text
samples x factors
```

For LSTM, planned, it will produce a 3D sequence tensor.

```text
samples x timesteps x factors
```

## Factor Builder

Creates reusable cross-domain factors from raw columns.

Factor families:

```text
lag
rolling
difference
ratio_to_rolling_mean
calendar
interaction, planned
technical indicators, planned
meteorological indicators, planned
```

Rainfall examples:

```text
rainfall_lag_1
rainfall_roll_6_sum
rainfall_roll_24_max
hour
season
```

Stock examples:

```text
close_lag_1
close_diff_5
close_ratio_roll_20_mean
day_of_week
```

## Experiment Generator

Expands presets into candidate experiments.

Example:

```text
rainfall_extreme_rain_lgbm_obs12_h3
rainfall_extreme_rain_lgbm_obs24_h3
rainfall_extreme_rain_lstm_obs24_h3
```

The generator is the foundation for automated experimentation. It lets SeqLens search target definitions, windows, factor combinations, and model families without human brainstorming every round.

## Automation Roadmap

Automation should progress in layers:

```text
rule-based planner
-> search planner
-> Optuna planner
-> LLM planner
```

The planner should output structured experiment configs only.

Allowed:

- propose target candidates
- propose factor sets
- propose observation windows
- choose model families
- stop when no validation improvement appears

Not allowed:

- edit arbitrary code
- optimize on test data
- bypass leakage checks
- hide failed experiments

## Model Roles

```text
naive / moving_average
  Minimum benchmarks.

LGBM
  Strong tabular factor model. Used to detect whether factors contain learnable signal.

LSTM
  Sequence model. Used after baselines and LGBM justify the added complexity.
```

## Rainfall Path

First rainfall task:

```text
Predict whether the next 3 hours have cumulative rainfall >= 80 mm.
```

Recommended first model:

```text
LGBMClassifier
```

Recommended metrics:

```text
recall
precision
f1
pr_auc
false_alarm_rate
miss_rate
```

## Stock Path

First stock task:

```text
Predict whether next-day close direction is positive.
```

Recommended first model:

```text
LGBMClassifier
```

Recommended metrics:

```text
f1
precision
recall
direction_accuracy
```

## Current MVP Boundary

Implemented:

- CSV loader
- time-series diagnostics
- time-based train / validation / test split
- baseline regression experiments
- baseline reports and comparison reports
- target builder
- factor builder
- window slicer
- rainfall and stock presets
- experiment candidate generator

Next:

- classification metrics
- event baseline
- LGBM optional plugin
- leakage report
- automated factor search

