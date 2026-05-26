# Rainfall Event Testing Notes

This note records the first external multi-station rainfall event test for SeqLens.
Rainfall is treated as a validation scenario for the cross-domain event workflow,
not as the core product boundary.

## Data Source

The test uses hourly station CSV files from the public `Raingel/historical_weather` repository, which documents its data as rebuilt from Taiwan CWA/CODiS station observations.

Raw URL pattern:

```text
https://raw.githubusercontent.com/Raingel/historical_weather/main/data/{station_id}/{station_id}_{year}.csv
```

## Stations

The first test used six Taiwan stations across different regions:

| Station ID | Name |
|---|---|
| 466920 | Taipei |
| 466940 | Keelung |
| 467490 | Taichung |
| 467530 | Alishan |
| 466990 | Hualien |
| 467660 | Taitung |

Years:

```text
2022, 2023
```

Cleaned columns:

```text
datetime
station_id
station_name
rainfall
```

Negative `Precp` values were clipped to `0` for this initial test.

## Event Definition

The event target is:

```text
future 3-hour cumulative rainfall >= threshold
```

Two thresholds were tested:

```text
40 mm
80 mm
```

The 80 mm threshold is closer to a short-duration extreme rainfall warning task. The 40 mm threshold is a broader strong-rain event task used to expose more positive samples.

## Initial Event Distribution

Input rows:

```text
105,120
```

Future 3-hour event counts before supervised feature dropping:

| Threshold | Events |
|---:|---:|
| 40 mm | 248 |
| 60 mm | 80 |
| 80 mm | 25 |
| 100 mm | 8 |

## Baseline Results

Model:

```text
event_majority
```

This baseline predicts the majority class from the training history.

### 80 mm Threshold

Supervised rows:

```text
104,958
```

Validation:

| Metric | Value |
|---|---:|
| Accuracy | 0.999 |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | 0.000 |
| False alarm rate | 0.000 |
| Miss rate | 1.000 |
| Positive support | 11 |

Test:

| Metric | Value |
|---|---:|
| Accuracy | 1.000 |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | 0.000 |
| False alarm rate | 0.000 |
| Miss rate | 1.000 |
| Positive support | 7 |

### 40 mm Threshold

Supervised rows:

```text
104,958
```

Validation:

| Metric | Value |
|---|---:|
| Accuracy | 0.996 |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | 0.000 |
| False alarm rate | 0.000 |
| Miss rate | 1.000 |
| Positive support | 84 |

Test:

| Metric | Value |
|---|---:|
| Accuracy | 0.997 |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | 0.000 |
| False alarm rate | 0.000 |
| Miss rate | 1.000 |
| Positive support | 64 |

## Problems Found

1. Extreme rainfall is highly imbalanced.

   At the 80 mm / 3h threshold, only 25 events appear before supervised feature dropping. A majority baseline gets near-perfect accuracy while missing all events.

2. Accuracy is misleading.

   Recall, precision, F1, and miss rate must be primary metrics for warning tasks.

3. The current event baseline is only a sanity check.

   It confirms class imbalance but is not a useful warning model.

4. The first runner needs stronger artifact naming for batch automation.

   This was fixed by adding unique suffixes when run directories collide in the same second.

5. External station files are not perfectly uniform.

   Some station/year combinations may be missing, so the data preparation layer needs availability checks.

6. The framework should keep rainfall-specific logic outside the core runner.

   Generic event concepts should be expressed through target definitions, rolling
   factors, threshold strategies, and entity-level validation artifacts.

## Generic Event-Aware Baseline

SeqLens now includes:

```text
recent_window_threshold
```

This baseline predicts an event when a configured rolling factor crosses a
threshold. For rainfall, this can be recent cumulative rainfall. In other
domains, the same baseline can represent recent return volatility, machine
sensor spikes, traffic load, or energy demand peaks.

Example:

```yaml
event_baseline:
  type: recent_window_threshold
  column: rainfall
  window: 3
  aggregation: sum
  threshold: 40
```

The matching rolling factor must exist in `factors.rolling`, for example:

```yaml
factors:
  rolling:
    columns: [rainfall]
    windows: [3]
    stats: [sum]
```

When automation tests several target thresholds, the baseline threshold follows
the active target threshold so candidates remain comparable.

### Auto-Run Check

The generic baseline was tested in an automated run with:

```text
thresholds: 40, 60
observation window: 12
models: event_majority, recent_window_threshold, lgbm
```

Key results:

| Threshold | Model | Validation Recall | Validation F1 | Test Recall | Test F1 |
|---:|---|---:|---:|---:|---:|
| 40 | event_majority | 0.000 | 0.000 | 0.000 | 0.000 |
| 40 | recent_window_threshold | 0.143 | 0.143 | 0.078 | 0.078 |
| 40 | lgbm, maximize_f1 | 0.500 | 0.091 | 0.016 | 0.026 |
| 40 | lgbm, maximize_recall | 0.512 | 0.073 | 0.172 | 0.025 |
| 60 | event_majority | 0.000 | 0.000 | 0.000 | 0.000 |
| 60 | recent_window_threshold | 0.000 | 0.000 | 0.031 | 0.031 |
| 60 | lgbm, maximize_f1 | 0.280 | 0.006 | 0.312 | 0.007 |

Interpretation:

- `recent_window_threshold` is a useful bridge between majority baselines and learned models.
- LGBM can increase recall, but sparse events make precision and false alarms unstable.
- The workflow should continue to compare simple rules, LGBM, and only later LSTM.

## Recommendations

1. Add an LGBM classifier plugin next.

   It should use the generated lag, rolling, and calendar factors to test whether the factors contain learnable signal.

   Status: implemented as an optional model via:

   ```bash
   pip install -e ".[lgbm]"
   seqlens run-event configs/rainfall_extreme_rain.yaml --model lgbm
   ```

## LGBM Classifier Test

After adding the optional LGBM plugin, the same six-station dataset was tested with:

```bash
seqlens run-event <threshold_config> --model lgbm
```

The model uses generated lag, rolling, and calendar factors. The decision threshold is selected on validation data by scanning thresholds from `0.05` to `0.95` and maximizing F1.

### 40 mm Threshold

Validation:

| Metric | Value |
|---|---:|
| Accuracy | 0.966 |
| Precision | 0.057 |
| Recall | 0.476 |
| F1 | 0.102 |
| False alarm rate | 0.032 |
| Miss rate | 0.524 |
| Positive support | 84 |

Test:

| Metric | Value |
|---|---:|
| Accuracy | 0.996 |
| Precision | 0.077 |
| Recall | 0.016 |
| F1 | 0.026 |
| False alarm rate | 0.001 |
| Miss rate | 0.984 |
| Positive support | 64 |

Top factors:

```text
hour
rainfall_roll_24_std
month
rainfall_roll_12_std
rainfall_roll_24_sum
rainfall_roll_6_std
```

### 80 mm Threshold

Validation:

| Metric | Value |
|---|---:|
| Accuracy | 0.950 |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | 0.000 |
| False alarm rate | 0.049 |
| Miss rate | 1.000 |
| Positive support | 11 |

Test:

| Metric | Value |
|---|---:|
| Accuracy | 0.850 |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | 0.000 |
| False alarm rate | 0.150 |
| Miss rate | 1.000 |
| Positive support | 7 |

Top factors:

```text
hour
rainfall_roll_24_std
month
rainfall_roll_12_std
rainfall_roll_24_sum
rainfall_lag_24
```

## Updated Problems Found

1. LGBM can learn some validation signal at the 40 mm threshold, but the signal does not generalize to the test split.

2. The 80 mm threshold has too few positive samples in 2022-2023 for a stable supervised experiment.

3. Calendar factors such as `hour` and `month` dominate feature importance. This may be legitimate seasonality, but it may also indicate that rainfall-only factors are too weak.

4. Validation threshold tuning by F1 is not enough. The model needs threshold strategies that optimize warning objectives, such as high recall with bounded false alarm rate.

5. The current split is global time-based across stations. Station-level distribution shifts should be reported before model comparison.

## Updated Recommendations

1. Expand the dataset period.

   Use at least 5-10 years for 80 mm / 3h extreme rainfall experiments.

2. Add station-level event distribution reports.

   Report positive support by station and split before training.

3. Add threshold search modes.

   Support:

   ```text
   maximize_f1
   maximize_recall
   maximize_recall_with_precision_floor
   minimize_miss_rate_with_false_alarm_cap
   ```

4. Add stronger rainfall event baselines.

   Useful baselines:

   ```text
   rolling_3h_sum_threshold
   rolling_6h_sum_threshold
   monthly_hourly_climatology
   ```

5. Add meteorological factors.

   Rainfall-only factors are likely insufficient. Add pressure, humidity, wind, and pressure-change features where station data supports them.

6. Add station-aware modeling.

   Options:

   ```text
   station one-hot encoding
   station-level models
   region preset
   leave-one-station-out validation
   ```

## Auto-Run Test

After adding the AutoExperimentRunner MVP, the same dataset was tested with:

```yaml
thresholds: [40, 60, 80]
observation_windows: [12, 24]
models: [event_majority, lgbm]
```

This produced 12 candidate experiments and wrote:

```text
event_distribution.csv
leaderboard.csv
recommendations.md
final_report.md
```

Best validation candidate:

```text
thr40_obs12_lgbm
```

Metrics:

| Split | Precision | Recall | F1 | False Alarm Rate | Positive Support |
|---|---:|---:|---:|---:|---:|
| Validation | 0.050 | 0.500 | 0.091 | 0.038 | 84 |
| Test | 0.077 | 0.016 | 0.026 | 0.001 | 64 |

Auto-run recommendations:

```text
- Some candidates have fewer than 30 validation positive events.
- Expand years, add stations, or lower threshold before promoting to LSTM.
- At least one LGBM candidate found non-zero test recall.
- Compare station-level event distribution before considering LSTM.
- Add threshold strategy modes beyond F1 maximization.
- Add rainfall-aware baselines such as rolling sum threshold.
- Add humidity, pressure, wind, and pressure-change factors when available.
```

Important note:

The current auto-runner already supports automated candidate expansion and comparison, but it should still be treated as an experiment screening layer. Its recommendations correctly indicate that the task is not ready for LSTM promotion yet.

## Station-Level Distribution Report

Auto-run now writes:

```text
event_distribution.csv
event_distribution.md
```

The report includes event support by:

```text
threshold
observation window
split
station
```

Example issue found in the six-station test:

```text
Some station-level splits have fewer than 5 positive events.
```

For the 40 mm / 3h task, validation examples showed sparse stations:

| Threshold | Observation | Split | Station |
|---:|---:|---|---|
| 40 | 12 | validation | 467660 |
| 40 | 12 | validation | 466940 |
| 40 | 12 | test | 466940 |

For the 80 mm / 3h task, the overall split support is already too small:

| Split | Positive Support |
|---|---:|
| Train | 7 |
| Validation | 11 |
| Test | 7 |

This confirms that the current data slice should not be promoted to LSTM. The next research step should improve event support and station balance first.

## Threshold Strategy Test

Auto-run now supports multiple threshold strategies for probability models:

```text
maximize_f1
maximize_recall
maximize_recall_with_precision_floor
minimize_miss_rate_with_false_alarm_cap
```

Test setup:

```yaml
thresholds: [40, 60]
observation_windows: [12]
models: [event_majority, lgbm]
threshold_strategies:
  - maximize_f1
  - maximize_recall
  - maximize_recall_with_precision_floor
  - minimize_miss_rate_with_false_alarm_cap
```

Key 40 mm / 3h results:

| Strategy | Validation Recall | Validation F1 | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|
| maximize_f1 | 0.500 | 0.091 | 0.077 | 0.016 | 0.026 |
| maximize_recall | 0.512 | 0.073 | 0.013 | 0.172 | 0.025 |
| maximize_recall_with_precision_floor | 0.500 | 0.091 | 0.077 | 0.016 | 0.026 |
| minimize_miss_rate_with_false_alarm_cap | 0.500 | 0.091 | 0.077 | 0.016 | 0.026 |

Interpretation:

```text
maximize_recall catches more test events, but precision collapses.
```

This means SeqLens should keep threshold strategy search as part of automated experimentation. For warning tasks, the best model depends on operational cost:

```text
High recall is useful when missed events are very costly.
Higher precision is useful when false alarms are expensive.
```

Current recommendation:

```text
Do not use a single default threshold strategy for all domains.
Let auto-run compare strategies and report the trade-off.
```

2. Add event-aware baselines.

   Useful baselines:

   ```text
   recent_rain_threshold
   rolling_sum_threshold
   climatology_by_month_hour
   ```

3. Add threshold search.

   Automatically compare:

   ```text
   40 mm, 60 mm, 80 mm, 100 mm
   ```

4. Add station-level reports.

   Event distribution varies by station. SeqLens should report positive support by station before training.

5. Add class imbalance handling.

   For LGBM:

   ```text
   class_weight
   scale_pos_weight
   threshold tuning
   PR-AUC
   ```

6. Add data availability checks.

   Before running experiments, report missing station/year files and skipped stations.

## Extreme Event Support Planning

SeqLens now writes:

```text
event_support_plan.csv
event_support_plan.md
```

This artifact scans threshold and horizon combinations before model promotion.
It is intended for rare or extreme event tasks where the original target may be
too sparse for stable validation.

Using the six-station rainfall sample, support improved substantially when the
future accumulation horizon was increased:

| Threshold | Horizon | Minimum Split Positive Support | Mean Event Rate |
|---:|---:|---:|---:|
| 40 | 3 | 64 | 0.0029 |
| 40 | 6 | 164 | 0.0078 |
| 40 | 12 | 386 | 0.0185 |
| 60 | 3 | 23 | 0.0010 |
| 60 | 6 | 93 | 0.0037 |
| 60 | 12 | 206 | 0.0094 |
| 80 | 3 | 7 | 0.0003 |
| 80 | 6 | 33 | 0.0017 |
| 80 | 12 | 119 | 0.0056 |

Interpretation:

- `80mm / 3h` is too sparse for reliable learned-model evaluation.
- `80mm / 6h` becomes barely usable for LGBM experimentation.
- `80mm / 12h` has enough support to test learned models more responsibly.
- Horizon planning is a cleaner first step than synthetic oversampling for rare events.
