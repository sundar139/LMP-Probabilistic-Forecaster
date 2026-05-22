# LMP Probabilistic Forecaster

Local-first probabilistic forecasting for PJM day-ahead hourly zonal LMPs.

## Current status

Implemented and validated through Step 7 foundation hardening:
- real AEP single-zone baseline workflow from Step 6 remains reproducible,
- optional MLflow tracking layer added (disabled by default),
- baseline training config hardened with stricter validation,
- rolling-origin backtest planning scaffold added (planning/report only).

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

## Backtest planning commands

Dry-run plan:

```bash
uv run python -m lmp_forecaster.cli plan-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24
```

Write plan report:

```bash
uv run python -m lmp_forecaster.cli plan-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24 \
  --write
```

## Generated paths and ignore policy

Generated artifacts remain local and ignored by Git:
- forecasts: `data/cache/forecasts/`
- reports: `data/cache/reports/`
- processed panel outputs: `data/processed/`
- model artifacts: `artifacts/`
- MLflow local tracking: `mlruns/`
- local runtime logs/checkpoints: `lightning_logs/`, `checkpoints/`

## Quality gates

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Next step

After this step, proceed to implementation of full rolling-origin training/evaluation execution and calibration-focused model improvements before hyperparameter tuning.
