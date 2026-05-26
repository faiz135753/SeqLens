# Rainfall Event Testing Notes

This note records the first external multi-station rainfall event test for SeqLens.

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
