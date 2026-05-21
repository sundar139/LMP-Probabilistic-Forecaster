# LMP Probabilistic Forecaster

A local-first, production-style probabilistic forecasting platform for PJM day-ahead hourly zonal LMPs. The project is designed for industry workflows with reproducible training and evaluation, multi-horizon uncertainty outputs (P10/P50/P90), rolling-origin backtesting, and serving interfaces for both APIs and interactive dashboards.

## Current implementation status

Foundation bootstrap, ingestion smoke layer, and cleaned single-zone panel builder are in place:

- environment, package layout, config system, source registry
- synthetic data generator and CLI harness
- smoke ingestion adapters for PJM + Open-Meteo
- leakage-safe single-zone panel builder (AEP-first)
- panel reporting utilities and mock-based tests

No model training is run in this step.

## Source coverage

- PJM Day-Ahead Hourly LMP Data Miner feed (`da_hrl_lmps`)
  - https://dataminer2.pjm.com/feed/da_hrl_lmps/definition
  - https://dataminer2.pjm.com/feed/da_hrl_lmps
- Open-Meteo Historical Weather API
  - https://open-meteo.com/en/docs/historical-weather-api
- Open-Meteo Historical Forecast API foundation
  - https://open-meteo.com/en/docs/historical-forecast-api

## Panel schema overview

Single-zone panel output is long-format and includes at minimum:
- core: `unique_id`, `ds`, `y`
- calendar features (hour/day/month/holiday + cyclic encodings)
- weather covariates (`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `apparent_temperature`, `precipitation`, `wind_speed_10m`, `cloud_cover`)
- leakage-safe lag features (`lmp_lag_1`, `lmp_lag_2`, `lmp_lag_3`, `lmp_lag_24`, `lmp_lag_48`, `lmp_lag_168`)
- leakage-safe rolling features (`lmp_rolling_mean_24`, `lmp_rolling_std_24`, `lmp_rolling_min_24`, `lmp_rolling_max_24`, `lmp_rolling_mean_168`, `lmp_rolling_std_168`)

## Leakage-safe feature policy

Lag and rolling features are computed per `unique_id` after sorting by `ds`.
Rolling features are built from `shift(1)` history first, so the current-row `y` cannot leak into its own feature value.

## Synthetic fallback policy

Synthetic fallback exists for pipeline smoke testing only:
- use `--allow-synthetic-lmp` when real cached LMP is unavailable
- use `--allow-synthetic-weather` when weather cache is unavailable

Synthetic output is explicitly labeled and should not be treated as final PJM training data.

## Tech stack

- Python 3.12 + uv (single managed environment)
- PyTorch (CUDA 12.8 index)
- NeuralForecast + Lightning
- Pandas/Polars/PyArrow/Numpy/Scikit-learn
- Optuna, MLflow, DVC
- FastAPI + Uvicorn, Streamlit
- Ruff, MyPy, PyTest

## Data policy

Raw PJM data and all PJM-derived artifacts must remain local and must not be committed to Git. This includes raw extracts, cache outputs, processed datasets, generated parquet/csv/json files, model checkpoints, experiment outputs, and report artifacts.

Tracked repository directories under `data/` only contain `.gitkeep` placeholders.

## Local setup

From PowerShell:

```powershell
cd "C:\Users\rohit\Documents\Personal Projects\LMP Probabilistic Forecaster"

# Install uv if missing
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv python install 3.12
uv python pin 3.12
uv sync
```

Or use the helper script:

```powershell
./scripts/bootstrap.ps1
```

## CLI smoke commands (dry-run)

These commands do not write files unless `--write` is provided:

```powershell
uv run python -m lmp_forecaster.cli list-sources
uv run python -m lmp_forecaster.cli list-zones
uv run python -m lmp_forecaster.cli pull-pjm-smoke --zone AEP
uv run python -m lmp_forecaster.cli pull-weather-smoke --zone AEP
uv run python -m lmp_forecaster.cli pull-historical-forecast-smoke --zone AEP
uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP
uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP --allow-synthetic-lmp --allow-synthetic-weather
```

## CLI write commands

Writes local artifacts under ignored paths only:

```powershell
uv run python -m lmp_forecaster.cli pull-pjm-smoke --zone AEP --write
uv run python -m lmp_forecaster.cli pull-weather-smoke --zone AEP --write
uv run python -m lmp_forecaster.cli pull-historical-forecast-smoke --zone AEP --write
uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP --allow-synthetic-lmp --allow-synthetic-weather --write --summary
```

## Test and quality checks

```powershell
uv run ruff check .
uv run mypy src
uv run pytest -q
```

or:

```powershell
./scripts/run_tests.ps1
```

Tests use mocked HTTP payloads and synthetic fixtures; they do not require live API calls.

## Troubleshooting

- PJM endpoint/query format changed:
  - Update request builder in `src/lmp_forecaster/data/pjm_lmp.py` (`build_day_ahead_lmp_request`).
- Rate limits / transient network failures:
  - Adjust retry/backoff in `src/lmp_forecaster/data/http_client.py`.
- Open-Meteo parameter changes:
  - Update request builders in `src/lmp_forecaster/data/openmeteo_weather.py`.
- Generic network failures:
  - Re-run in dry-run mode first to verify params and output paths.
- Windows `PYTHONHOME` contamination:
  - Clear `PYTHONHOME` and `PYTHONPATH` in the shell/session before running `uv run ...`.

## Next implementation step

Train single-zone TFT baseline and DeepAR benchmark using the cleaned panel.
