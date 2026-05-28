# Bike Sharing Positive Benchmark

This benchmark uses the UCI Bike Sharing Dataset as a high-signal, cross-domain
event prediction task. It is intended to prove that SeqLens can discover stable
predictive signal when the data has strong seasonality, daily cycles, weather
context, and enough positive event support.

## Dataset

- Source: UCI Bike Sharing Dataset
- Granularity: hourly
- Period: 2011-2012
- Rows: 17,379
- Main target column: `demand`
- Context columns: weather situation, temperature, humidity, wind speed,
  holiday, working day, weekday, hour, month, and season

Prepare the data:

```bash
python scripts/prepare_bike_sharing_sample.py
```

Run the automated event experiment:

```bash
seqlens auto-run configs/bike_sharing_high_demand.yaml
```

## Prediction Task

The benchmark predicts whether the next hour enters a high-demand regime:

```text
past hourly demand, weather, and calendar factors -> next-hour high demand event
```

The automated run scans demand thresholds of `300`, `400`, and `500`, observation
windows of `24` and `48` hours, simple event-aware baselines, and LGBM threshold
strategies.

## Reference Result

An initial local run produced the following best LGBM test results:

| Event threshold | Precision | Recall | F1 | Accuracy | False alarm rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| demand >= 300 | 0.915 | 0.922 | 0.918 | 0.944 | 0.044 |
| demand >= 400 | 0.837 | 0.966 | 0.897 | 0.951 | 0.053 |
| demand >= 500 | 0.771 | 0.956 | 0.854 | 0.953 | 0.047 |

The best simple `recent_window_threshold` baseline reached F1 scores of `0.585`,
`0.380`, and `0.233` for the same thresholds. This makes the benchmark useful as
a positive control: LGBM should clearly outperform simple rules.

## Why This Benchmark Matters

SeqLens also includes difficult extreme-event scenarios such as heavy rainfall
and stock downside risk. Those tasks often need external signals, careful event
support planning, and false-alarm constraints. The bike sharing task serves the
opposite role: it is a domain where the automated workflow should succeed. A
healthy SeqLens release should be able to:

- mark event support as usable or normal,
- promote LGBM,
- recommend LSTM promotion when LGBM is stable,
- show LGBM outperforming simple event rules,
- report interpretable factors such as recent demand changes, hour-of-day, and
  rolling demand patterns.
