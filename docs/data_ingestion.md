# Data ingestion smoke layer

## Source overview

This project includes smoke-test ingestion adapters for:

- PJM Day-Ahead Hourly LMP feed (`da_hrl_lmps`)
  - Feed definition: https://dataminer2.pjm.com/feed/da_hrl_lmps/definition
  - Endpoint: https://dataminer2.pjm.com/feed/da_hrl_lmps
- Open-Meteo Historical Weather API
  - https://open-meteo.com/en/docs/historical-weather-api
- Open-Meteo Historical Forecast API (foundation)
  - https://open-meteo.com/en/docs/historical-forecast-api

The historical forecast adapter is included now so later backtests can use weather information that would have been available at prediction time (leakage-free design).

## Why PJM raw data is not committed

Raw PJM values and PJM-derived artifacts are private operational data. Repository policy keeps only code/config/tests/docs in Git. Any generated pulls must stay in ignored local cache paths.

## Cache layout

All smoke outputs are local-only and ignored by Git:

- `data/cache/pjm/da_hrl_lmps/*.parquet`
- `data/cache/weather/openmeteo/*.parquet`
- optional raw payload dumps under cache are also ignored.

## Smoke ingestion workflow

1. Run dry-runs first (no writes, prints planned request/path).
2. Use `--write` only when you want local cache artifacts.
3. Keep date ranges small and smoke-oriented.

CLI commands:

- `uv run python -m lmp_forecaster.cli list-sources`
- `uv run python -m lmp_forecaster.cli list-zones`
- `uv run python -m lmp_forecaster.cli pull-pjm-smoke --zone AEP`
- `uv run python -m lmp_forecaster.cli pull-weather-smoke --zone AEP`
- `uv run python -m lmp_forecaster.cli pull-historical-forecast-smoke --zone AEP`

Write mode examples:

- `uv run python -m lmp_forecaster.cli pull-pjm-smoke --zone AEP --write`
- `uv run python -m lmp_forecaster.cli pull-weather-smoke --zone AEP --write`
- `uv run python -m lmp_forecaster.cli pull-historical-forecast-smoke --zone AEP --write`

PowerShell helper:

- `scripts/pull_smoke_data.ps1` runs dry-runs only and prints exact write commands.

## Validation checks

Validation helpers enforce:

- required columns and null checks
- monotonic hourly timestamps per series
- expected one-hour spacing
- DST-aware day-length detection (23/24/25 hour days)

## Known limitations

- PJM Data Miner query parameters can evolve; request builder is centralized for easy adjustment.
- PJM payload column names can vary; normalization uses candidate-matching with explicit errors if LMP fields are missing.
- Historical forecast endpoint behavior can change; adapter is scaffolding for future leakage-safe strictness.
- Unit tests are mock-based and intentionally do not require live network access.
