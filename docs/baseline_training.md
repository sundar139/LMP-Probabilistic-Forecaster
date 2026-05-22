# Baseline training and evaluation status

## Status

- Real AEP single-zone baseline workflow is in place and reproducible.
- Current TFT/DeepAR metrics are first-run untuned metrics and are not final benchmark claims.
- Coverage is below desired calibration range for production confidence:
  - TFT coverage_80 currently below the 70%-90% guide target.
  - DeepAR coverage_80 is currently 0.0 (interval under-coverage/collapse behavior).
- Next major objective is reliable rolling-origin evaluation before tuning.

## Commands

Dry-run training (no writes):

```bash
uv run python -m lmp_forecaster.cli train-single-zone-baselines --zone AEP
```

Write training without MLflow:

```bash
uv run python -m lmp_forecaster.cli train-single-zone-baselines \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --write
```

Write training with MLflow tracking enabled:

```bash
uv run python -m lmp_forecaster.cli train-single-zone-baselines \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --enable-tracking \
  --experiment-name lmp_probabilistic_forecaster_smoke \
  --tracking-uri file:./mlruns \
  --write
```

Dry-run rolling-origin backtest planning:

```bash
uv run python -m lmp_forecaster.cli plan-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24
```

Write rolling-origin backtest plan reports:

```bash
uv run python -m lmp_forecaster.cli plan-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24 \
  --write
```

## Training configuration hardening

Main training config keys are explicit and comparable under `conf/training.yaml`:

- `horizon_hours: 24`
- `input_size_hours: 168`
- `quantiles: [0.1, 0.5, 0.9]`
- `interval_level: 80`
- `validation_hours: 72`
- `test_hours: 72`
- `seed`
- `max_steps_smoke`
- `max_steps_real_candidate`
- `batch_size`
- `accelerator` (`auto|gpu|cpu`)

Validation hardening now rejects:
- unsorted/invalid quantiles,
- interval-level mismatch vs quantile span,
- non-positive split sizes,
- invalid accelerator values.

## Forecast schema contract

Forecast normalization/validation enforces:
- `unique_id`, `ds`, `y`,
- `p10`, `p50`, `p90`,
- `model`, `generated_at`,
- `data_source_label`, `zone`.

And checks:
- `p10 <= p50 <= p90`,
- no NaN in quantiles,
- actual `y` present for evaluation.

## Experiment tracking (MLflow)

Tracking is optional and disabled by default.

Config file: `conf/tracking.yaml`

Supported keys:
- `enabled`
- `experiment_name`
- `tracking_uri`
- `run_name_prefix`
- `log_artifacts`
- `log_model_artifacts`

Default local URI:
- `file:./mlruns`

Safety behavior:
- If tracking is disabled or unavailable, training still runs normally.
- Secret-like params (keys containing token/key/secret/password/api/credential) are skipped from tracking logs.
- Artifact logging records reference paths/metadata, not private secrets.

## Rolling-origin planning scaffold

Module: `src/lmp_forecaster/eval/backtest.py`

Includes:
- `BacktestConfig`
- `BacktestFold`
- `make_rolling_origin_folds()`
- `validate_backtest_folds()`
- `summarize_backtest_plan()`
- `write_backtest_plan()`

Supported planning modes:
- `expanding`
- `rolling`

Fold checks enforce:
- `train_end < test_start`
- leakage check before origin
- no overlapping test windows
- minimum training history per fold
- timezone preservation when present

This step plans folds only; it does not run full fold-by-fold model training.

## Artifact locations and ignore policy

Generated outputs remain local and gitignored:
- forecasts: `data/cache/forecasts/`
- reports: `data/cache/reports/`
- baseline artifacts: `artifacts/baselines/`
- tracking runs: `mlruns/`

## Interpretation note

Current metrics indicate:
- TFT currently outperforms DeepAR on MAE/RMSE for this untuned run.
- Both models still require calibration/tuning.
- DeepAR under-coverage should be treated as a model/config issue to fix, not as a final performance conclusion.
