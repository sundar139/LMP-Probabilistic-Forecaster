# Panel building

## Purpose

The single-zone panel is the training-ready long-format dataset for early model baselines. It combines hourly target values (`y`) with leakage-safe features and aligned weather covariates for one zone at a time.

## Input schema

LMP input must include:
- `unique_id`
- `ds`
- `y`

Weather input should include:
- `ds`
- `temperature_2m`
- `relative_humidity_2m`
- `dew_point_2m`
- `apparent_temperature`
- `precipitation`
- `wind_speed_10m`
- `cloud_cover`
- optional `source`

## Output schema

Output includes:
- required panel core: `unique_id`, `ds`, `y`
- calendar: hour/day/month/year flags + cyclic encodings
- weather columns listed above
- weather missingness indicators
- lags: `lmp_lag_1`, `lmp_lag_2`, `lmp_lag_3`, `lmp_lag_24`, `lmp_lag_48`, `lmp_lag_168`
- rolling: `lmp_rolling_mean_24`, `lmp_rolling_std_24`, `lmp_rolling_min_24`, `lmp_rolling_max_24`, `lmp_rolling_mean_168`, `lmp_rolling_std_168`

## Calendar features

Calendar features are generated in `src/lmp_forecaster/features/calendar.py` using `America/New_York` timezone and US holidays (`holidays` package).

## Weather features

Weather is normalized and aligned in `src/lmp_forecaster/features/weather.py`.
- default alignment is exact hourly join on `ds`
- nearest-hour join is optional and off by default
- optional controlled fill applies to weather only
- target `y` is never filled

## Leakage-safe lag and rolling logic

`src/lmp_forecaster/features/lags.py` applies lags per `unique_id` after sorting by `ds`.
Rolling features are computed from `shift(1)` first, then rolling windows. This prevents current-row target leakage.

## DST/timezone policy

- Panel timestamps are localized/converted to `America/New_York`.
- DST behavior is preserved via tz-aware timestamps.
- Reporting includes DST day-length summaries where relevant.

## Synthetic fallback policy

If real cached inputs are unavailable:
- synthetic LMP fallback is allowed only with `--allow-synthetic-lmp`
- synthetic weather fallback is allowed only with `--allow-synthetic-weather`

Synthetic outputs are clearly labeled for smoke/pipeline checks and are not a substitute for real PJM training data.

## Validation checks

Panel validation enforces:
- required columns
- no nulls in `unique_id`, `ds`, `y`
- monotonic sorted hourly order per series
- no duplicate `unique_id + ds`
- `y` numeric
- lag leakage check passes
- warmup rows dropped when configured so lag/rolling nulls are removed

## Example commands

Dry-run:
- `uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP`
- `uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP --allow-synthetic-lmp --allow-synthetic-weather`

Write output + summary:
- `uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP --allow-synthetic-lmp --allow-synthetic-weather --write --summary`
