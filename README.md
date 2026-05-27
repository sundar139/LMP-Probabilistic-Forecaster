# LMP Probabilistic Forecaster

Local-first probabilistic forecasting for PJM day-ahead hourly zonal LMPs.

## Current status

Implemented and validated through portable tuning package and candidate import validation closeout:
- real AEP single-zone baseline workflow remains reproducible,
- optional MLflow tracking layer remains available (disabled by default),
- real rolling-origin AEP backtest execution completed for TFT and DeepAR across 3 folds,
- calibration diagnostics summarize coverage/width/crossing/bias from rolling forecasts,
- focused TFT/DeepAR search design generates evidence-driven search spaces,
- focused tuning execution supports local resource-safe mode with explicit heavy-run guard,
- portable tuning package export supports `local_safe`, `cloud_16gb`, and `cloud_24gb` profiles,
- imported ranked candidates are revalidated locally with promotion recompute and mismatch detection,
- full heavy tuning remains deferred on this local machine due to hardware limits.

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

## Portable tuning package and import validation workflow

Why full tuning is not run locally:
- this workstation is constrained to roughly 8GB VRAM, 16GB RAM, and around 100GB free disk,
- earlier heavier tuning attempts were unstable,
- local execution is intentionally limited to planning and bounded validation.

Resource profiles:
- `local_safe`: tiny local planning/smoke profile,
- `cloud_16gb`: intended first external package target,
- `cloud_24gb`: larger external search profile.

Export package dry-run (no writes):

```bash
uv run python -m lmp_forecaster.cli export-tuning-package \
  --zone AEP \
  --resource-profile cloud_16gb \
  --models TFT,DeepAR \
  --max-trials 12 \
  --folds 2
```

Export package write:

```bash
uv run python -m lmp_forecaster.cli export-tuning-package \
  --zone AEP \
  --resource-profile cloud_16gb \
  --models TFT,DeepAR \
  --max-trials 12 \
  --folds 2 \
  --write
```

Import external ranked results and recompute promotion locally:

```bash
uv run python -m lmp_forecaster.cli import-tuning-results \
  --zone AEP \
  --ranked-candidates-path <ranked_candidates_csv>
```

Write import validation report:

```bash
uv run python -m lmp_forecaster.cli import-tuning-results \
  --zone AEP \
  --ranked-candidates-path <ranked_candidates_csv> \
  --write
```

Import behavior summary:
- required candidate schema is enforced,
- promotion decisions are recomputed locally from baseline metrics,
- imported promotion labels are treated as advisory only,
- mismatch between imported label and recomputed status is reported,
- under-covered or interval-collapse candidates are rejected.

If no candidate is promoted:
- keep current baseline as active,
- tune externally with stronger profile (`cloud_16gb` or `cloud_24gb`),
- import new ranked results,
- re-run 3-fold rolling backtest for the best candidate only before any promotion.
