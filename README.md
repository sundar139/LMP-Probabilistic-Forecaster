# LMP Probabilistic Forecaster

A local-first, production-style probabilistic forecasting platform for PJM day-ahead hourly zonal LMPs. The project is designed for industry workflows with reproducible training and evaluation, multi-horizon uncertainty outputs (P10/P50/P90), rolling-origin backtesting, and serving interfaces for both APIs and interactive dashboards.

## Current implementation status

Foundation bootstrap, ingestion smoke layer, cleaned single-zone panel builder, and single-zone baseline training are in place:

- environment, package layout, config system, source registry
- synthetic data generator and CLI harness
- smoke ingestion adapters for PJM + Open-Meteo
- leakage-safe single-zone panel builder (AEP-first)
- single-zone TFT + DeepAR baseline training CLI
- panel/reporting/metrics utilities and mock-based tests

No multi-zone training or rolling-origin backtesting is run in this step.

## Source coverage

- PJM Day-Ahead Hourly LMP Data Miner feed (`da_hrl_lmps`)
  - https://dataminer2.pjm.com/feed/da_hrl_lmps/definition
  - https://dataminer2.pjm.com/feed/da_hrl_lmps
- Open-Meteo Historical Weather API
  - https://open-meteo.com/en/docs/historical-weather-api
- Open-Meteo Historical Forecast API foundation
  - https://open-meteo.com/en/docs/historical-forecast-api

## Baseline models

- TFT baseline (probabilistic)
- DeepAR benchmark (probabilistic)

Both baselines produce quantile outputs:
- P10
- P50
- P90

## Synthetic smoke training vs real PJM training

Synthetic fallback is supported for smoke/pipeline validation when real cached panel inputs are unavailable.

Important: synthetic smoke metrics are not project performance claims and must not be presented as real PJM results.

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

Or use helper scripts:

```powershell
./scripts/bootstrap.ps1
./scripts/build_panel_smoke.ps1
./scripts/train_baseline_smoke.ps1
```

## Dry-run commands

```powershell
uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP
uv run python -m lmp_forecaster.cli build-single-zone-panel --zone AEP --allow-synthetic-lmp --allow-synthetic-weather
uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP
uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP --allow-synthetic-panel --build-panel-if-missing
```

## Smoke write command

```powershell
uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP --allow-synthetic-panel --build-panel-if-missing --write
```

## Expected artifact locations

- panel parquet: `data/processed/panel/single_zone/`
- forecast parquet: `data/cache/forecasts/`
- metrics/report JSON/CSV: `data/cache/reports/`
- baseline artifacts: `artifacts/baselines/`

All are ignored by Git.

## Warning on smoke metrics

Metrics produced from synthetic panels are smoke-test diagnostics only. They are not final project results.

## Test and quality checks

```powershell
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Troubleshooting

- PJM endpoint/query format changed:
  - Update request builder in `src/lmp_forecaster/data/pjm_lmp.py` (`build_day_ahead_lmp_request`).
- NeuralForecast API/version mismatches:
  - Inspect installed signatures and adapt constructor args in `src/lmp_forecaster/models/baselines.py`.
- Rate limits / transient network failures:
  - Adjust retry/backoff in `src/lmp_forecaster/data/http_client.py`.
- Open-Meteo parameter changes:
  - Update request builders in `src/lmp_forecaster/data/openmeteo_weather.py`.
- Windows `PYTHONHOME` contamination:
  - Clear `PYTHONHOME` and `PYTHONPATH` before `uv run ...`.

## Next implementation step

Fix PJM live endpoint/query format and connect real cached AEP LMP data into the panel/training path.
