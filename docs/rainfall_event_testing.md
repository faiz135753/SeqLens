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

