# Real baseline training (AEP, 2024)

This document records the first real single-zone untuned baseline run for AEP.

## 1) Real data used

LMP source:
- PJM day-ahead hourly LMP (`da_hrl_lmps`), zone `AEP`
- normalized local cache under `data/cache/pjm/da_hrl_lmps/`
- quality report: `data/cache/reports/lmp_quality_AEP_20260522T200356Z.json`
- 2024 row_count: 8784 (expected 8784)
- missing_hour_count: 0
- duplicate_timestamp_count: 0
- timezone: America/New_York

Weather source:
- Open-Meteo archive API
- cache file: `data/cache/weather/openmeteo/openmeteo_AEP_2024-01-01_2024-12-31.parquet`
- quality report: `data/cache/reports/weather_quality_AEP_20260522T201453Z.json`
- row_count: 8784
- missing_hour_count: 0
- duplicate_timestamp_count: 0
- timezone: America/New_York
- source_label: real/openmeteo

## 2) Weather covariates

Required hourly variables:
- temperature_2m
- relative_humidity_2m
- dew_point_2m
- apparent_temperature
- precipitation
- wind_speed_10m
- cloud_cover

Observed 2024 missing values (all zero):
- temperature_2m: 0
- relative_humidity_2m: 0
- dew_point_2m: 0
- apparent_temperature: 0
- precipitation: 0
- wind_speed_10m: 0
- cloud_cover: 0

## 3) Panel construction

Command:
```bash
uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP --start-date 2024-01-01 --end-date 2024-12-31 --write --summary
```

Output:
- panel: `data/processed/panel/single_zone/AEP_panel.parquet`
- summary: `data/cache/reports/panel_summary_AEP_20260522T200531Z.json`

Construction highlights:
- real LMP + real weather only
- warmup rows dropped (`min_history_hours=168`)
- lag and rolling features use shifted target only (no leakage)
- target `y` never interpolated
- source labels retained (`source_label=real`, `data_source_label=real`)

Panel result:
- row_count: 8616
- start_ds: 2024-01-08 00:00:00-05:00
- end_ds: 2024-12-31 23:00:00-05:00
- duplicate timestamps: 0
- missing_target_values: 0

## 4) Split policy

Chronological fixed split from panel tail:
- train: 8472 rows
- validation: 72 rows
- test: 72 rows

No overlap and no shuffling.

## 5) TFT and DeepAR config

Training command:
```bash
uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP --panel-path data/processed/panel/single_zone/AEP_panel.parquet --write
```

Shared configuration:
- horizon: 24
- input_size: 168
- quantiles: (0.1, 0.5, 0.9)
- deterministic seed: 42
- smoke max_steps: 30
- data_source_label requirement: real

Models:
- TFT baseline (`neuralforecast.models.TFT`)
- DeepAR baseline (`neuralforecast.models.DeepAR`)

Accelerator used:
- gpu (`NVIDIA GeForce RTX 4070 Laptop GPU`)

## 6) Probabilistic forecast schema

Required schema validated for both model outputs:
- unique_id
- ds
- y
- p10
- p50
- p90
- model
- generated_at
- data_source_label
- zone

Validation checks:
- `p10 <= p50 <= p90`
- no NaN in p10/p50/p90
- `y` present for evaluation
- `data_source_label=real`
- `zone=AEP`

Forecast outputs:
- `data/cache/forecasts/tft_forecast_AEP.parquet`
- `data/cache/forecasts/deepar_forecast_AEP.parquet`

## 7) Metrics and formulas

Metrics reported per model:
- MAE
- RMSE
- pinball_p10
- pinball_p50
- pinball_p90
- mean_pinball_loss
- coverage_80
- interval_width_mean

Definitions:
- pinball at quantile q: `mean(max(q*(y-yhat), (q-1)*(y-yhat)))`
- mean_pinball_loss: arithmetic mean of p10/p50/p90 pinball losses
- coverage_80: fraction where `y` in `[p10, p90]`
- interval_width_mean: mean of `(p90 - p10)`

## 8) First real results

From `data/cache/reports/baseline_metrics_AEP.json`:

| model | row_count | MAE | RMSE | pinball_p10 | pinball_p50 | pinball_p90 | mean_pinball_loss | coverage_80 | interval_width_mean | data_source_label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TFT | 72 | 5.611238 | 6.605262 | 1.534832 | 2.805619 | 1.259538 | 1.866663 | 0.555556 | 13.402272 | real |
| DeepAR | 72 | 21.906022 | 22.495103 | 2.433182 | 10.953011 | 17.357774 | 10.247989 | 0.000000 | 5.045403 | real |

Related reports:
- `data/cache/reports/baseline_training_report_AEP.json`
- `data/cache/reports/baseline_results_summary_AEP_20260522T201339Z.json`

## 9) Limitations

- single zone only (AEP)
- untuned baseline configuration
- one fixed chronological split
- no rolling-origin backtest yet
- no Optuna tuning yet

## 10) Next steps

Step 7 focus:
- tighten real baseline training configuration
- add MLflow tracking
- prepare rolling-origin backtest design
