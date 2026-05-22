# LMP Probabilistic Forecaster

Local-first probabilistic forecasting for PJM day-ahead hourly zonal LMPs.

## Current status

Implemented and validated through Step 6 continuation with the first real AEP single-zone baseline workflow:
- real PJM AEP ingestion cache validation
- real Open-Meteo weather backfill and quality checks
- real single-zone panel build (with warmup drop)
- real TFT and DeepAR untuned baseline training

First real single-zone untuned baseline, not final tuned benchmark.

## Real AEP PJM LMP status (2024)

Source: PJM `da_hrl_lmps` normalized cache
- row_count: 8784
- expected_hour_count: 8784
- missing_hour_count: 0
- duplicate_timestamp_count: 0
- min_ds: 2024-01-01 00:00:00-05:00
- max_ds: 2024-12-31 23:00:00-05:00
- data_source_label: real

Reference report:
- `data/cache/reports/lmp_quality_AEP_20260522T200356Z.json`

## Real weather status (AEP, 2024)

Source: Open-Meteo archive API (AEP proxy coordinates)
- window: 2024-01-01 to 2024-12-31
- row_count: 8784
- timezone: America/New_York
- missing values:
  - temperature_2m: 0
  - relative_humidity_2m: 0
  - dew_point_2m: 0
  - apparent_temperature: 0
  - precipitation: 0
  - wind_speed_10m: 0
  - cloud_cover: 0
- source_label: real/openmeteo
- data_source_label: real

Reference report:
- `data/cache/reports/weather_quality_AEP_20260522T201453Z.json`

## Real panel status

Panel output:
- `data/processed/panel/single_zone/AEP_panel.parquet`

Summary:
- row_count after warmup: 8616
- start_ds: 2024-01-08 00:00:00-05:00
- end_ds: 2024-12-31 23:00:00-05:00
- duplicate timestamps: 0
- null target values: 0
- source_labels: [real]

Feature groups included:
- target and keys: `unique_id`, `ds`, `y`
- weather covariates + missingness flags
- calendar/time features
- lag features (`lag_1`, `lag_2`, `lag_3`, `lag_24`, `lag_48`, `lag_168`)
- rolling features built from shifted target only
- source labels (`source_label`, `data_source_label`)

Reference report:
- `data/cache/reports/panel_summary_AEP_20260522T200531Z.json`

## First real baseline metrics (AEP)

| model | row_count | MAE | RMSE | pinball_p10 | pinball_p50 | pinball_p90 | mean_pinball_loss | coverage_80 | interval_width_mean | data_source_label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TFT | 72 | 5.6112 | 6.6053 | 1.5348 | 2.8056 | 1.2595 | 1.8667 | 0.5556 | 13.4023 | real |
| DeepAR | 72 | 21.9060 | 22.4951 | 2.4332 | 10.9530 | 17.3578 | 10.2480 | 0.0000 | 5.0454 | real |

Primary report files:
- `data/cache/reports/baseline_metrics_AEP.json`
- `data/cache/reports/baseline_training_report_AEP.json`
- `data/cache/reports/baseline_results_summary_AEP_20260522T201339Z.json`

## Reproduction commands

```bash
# 1) Pull real weather cache (write)
uv run python -m lmp_forecaster.cli pull-real-weather --zone AEP --start-date 2024-01-01 --end-date 2024-12-31 --write

# 2) Build real single-zone panel (write + summary)
uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP --start-date 2024-01-01 --end-date 2024-12-31 --write --summary

# 3) Train real untuned baselines (write)
uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP --panel-path data/processed/panel/single_zone/AEP_panel.parquet --write
```

## Quality gates

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Next step

Step 7: tighten real baseline training configuration, add MLflow tracking, and prepare rolling-origin backtest design.
