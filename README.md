# LMP Probabilistic Forecaster

Local-first probabilistic forecasting for PJM day-ahead hourly zonal LMPs.

## Current status

Implemented and validated through Step 8 foundation pass:
- real AEP single-zone baseline workflow from Step 6 remains reproducible,
- optional MLflow tracking layer added (disabled by default),
- baseline training config hardened with stricter validation,
- rolling-origin planning scaffold from Step 7 is now extended to real fold execution,
- real rolling-origin AEP backtest execution completed for TFT and DeepAR across 3 folds.

Current real AEP metrics are first-run untuned metrics and are not final benchmark claims.

## Real AEP baseline evidence snapshot

From existing Step 6 evidence:
- real PJM LMP rows (2024): 8784
- missing hours: 0
- duplicate timestamps: 0
- real weather rows (2024): 8784
- real panel rows after warmup: 8616

Baseline caveat:
- TFT currently outperforms DeepAR on MAE/RMSE in this untuned run.
- TFT coverage_80 is below the desired 70%-90% guide range.
- DeepAR coverage_80 is 0.0 and should be treated as a calibration/config issue, not final model quality.

## Training commands

Dry-run training (no writes):

```bash
uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP
```

Write training without tracking:

```bash
uv run python -m lmp_forecaster.cli train-single-zone-baselines \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --write
```

Write training with MLflow enabled:

```bash
uv run python -m lmp_forecaster.cli train-single-zone-baselines \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --enable-tracking \
  --experiment-name lmp_probabilistic_forecaster_smoke \
  --tracking-uri file:./mlruns \
  --write
```

## Backtest execution commands

Dry-run execution (plan only, no training/writes/tracking):

```bash
uv run python -m lmp_forecaster.cli run-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24
```

Real write execution:

```bash
uv run python -m lmp_forecaster.cli run-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24 \
  --write
```

Useful toggles:
- `--skip-tft` or `--skip-deepar` for single-model runs.
- `--enable-tracking --tracking-uri <uri> --experiment-name <name>` for optional MLflow logging.
- `--max-steps <int>` to cap per-fold smoke-safe training steps.

## Generated paths and ignore policy

Generated artifacts remain local and ignored by Git:
- backtest forecasts/metrics: `data/cache/backtests/`
- reports: `data/cache/reports/`
- forecast caches: `data/cache/forecasts/`
- processed panel outputs: `data/processed/`
- model artifacts: `artifacts/`
- MLflow local tracking: `mlruns/`
- MLflow scratch artifacts: `.mlflow_artifacts/`
- local runtime logs/checkpoints: `lightning_logs/`, `checkpoints/`

## Quality gates

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Next step

Proceed to Step 9: calibration and focused hyperparameter search design for TFT/DeepAR using rolling-origin evidence.
