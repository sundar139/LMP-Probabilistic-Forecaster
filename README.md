# LMP Probabilistic Forecaster

Local-first probabilistic forecasting for PJM day-ahead hourly zonal LMPs.

## Current status

Implemented and validated through Step 10 resource-safe focused tuning closeout:
- real AEP single-zone baseline workflow from Step 6 remains reproducible,
- optional MLflow tracking layer remains available (disabled by default),
- real rolling-origin AEP backtest execution completed for TFT and DeepAR across 3 folds,
- calibration diagnostics summarize coverage/width/crossing/bias from rolling forecasts,
- focused TFT/DeepAR search design generates evidence-driven search spaces,
- focused tuning execution now supports local resource-safe mode with explicit heavy-run guard,
- local write-mode smoke tuning completed with bounded settings (2 trials, 1 fold, max_steps_cap 3),
- full 12-trial first pass remains deferred on this local machine due to hardware stability limits.

Current real AEP metrics and smoke tuning diagnostics are workflow evidence, not final benchmark claims.

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

## Calibration diagnostics and focused search design commands

Calibration diagnostics dry-run (reads latest rolling outputs, writes nothing):

```bash
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP
```

Calibration diagnostics write-mode:

```bash
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP --write
```

Focused search design dry-run:

```bash
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP
```

Focused search design write-mode:

```bash
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP --write
```

Rationale:
- TFT remains under-covered and needs calibration-oriented adjustments.
- DeepAR shows interval collapse behavior and needs targeted distribution/calibration recovery.
- This step produces design artifacts only; it does not run large tuning.

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Focused tuning execution (resource-safe local mode)

Dry-run resource-safe mode:

```bash
uv run python -m lmp_forecaster.cli run-focused-tuning \
  --zone AEP \
  --resource-profile local_safe \
  --max-trials 2 \
  --folds 1 \
  --max-steps-cap 3 \
  --skip-deepar
```

Write-mode resource-safe smoke run:

```bash
uv run python -m lmp_forecaster.cli run-focused-tuning \
  --zone AEP \
  --resource-profile local_safe \
  --max-trials 2 \
  --folds 1 \
  --max-steps-cap 3 \
  --skip-deepar \
  --write
```

Behavior summary:
- local-safe defaults: small batch size, `num_workers=0`, tiny step cap, cleanup-after-trial on,
- heavy run refused unless `--allow-heavy-run` is explicitly provided,
- MLflow is optional and disabled by default (`--no-mlflow` supported),
- full first pass (`max_trials_first_pass=12`, `folds_for_full_first_pass=2`) is documented but deferred locally.

Smoke tuning interpretation:
- smoke results validate workflow reliability under hardware constraints,
- smoke results do not justify final promotion without multi-fold validation on stronger hardware.
