# LMP Probabilistic Forecaster

A local-first, production-style probabilistic forecasting platform for PJM day-ahead hourly zonal LMPs. The project is designed for industry workflows with reproducible training and evaluation, multi-horizon uncertainty outputs (P10/P50/P90), rolling-origin backtesting, and serving interfaces for both APIs and interactive dashboards.

## Current implementation status

Foundation bootstrap and ingestion smoke layer are in place:

- environment, package layout, config system, source registry
- synthetic data generator and CLI harness
- smoke ingestion adapters for PJM + Open-Meteo
- validation utilities and mock-based ingestion tests

No model training is run in this step.

## Source coverage

- PJM Day-Ahead Hourly LMP Data Miner feed (`da_hrl_lmps`)
  - https://dataminer2.pjm.com/feed/da_hrl_lmps/definition
  - https://dataminer2.pjm.com/feed/da_hrl_lmps
- Open-Meteo Historical Weather API
  - https://open-meteo.com/en/docs/historical-weather-api
- Open-Meteo Historical Forecast API foundation
  - https://open-meteo.com/en/docs/historical-forecast-api

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
```

## CLI smoke commands (write mode)

Writes local cache artifacts under ignored paths only:

```powershell
uv run python -m lmp_forecaster.cli pull-pjm-smoke --zone AEP --write
uv run python -m lmp_forecaster.cli pull-weather-smoke --zone AEP --write
uv run python -m lmp_forecaster.cli pull-historical-forecast-smoke --zone AEP --write
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

Tests use mocked HTTP payloads and do not require live API calls.

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

Build a cleaned single-zone panel by combining cached LMP + weather + calendar + lag features.
