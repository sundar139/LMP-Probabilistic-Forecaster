# Baseline training and evaluation status

## Status

- Real AEP single-zone baseline workflow remains reproducible.
- Rolling-origin AEP backtest execution is now implemented and has been run for TFT + DeepAR across 3 real folds.
- Current TFT/DeepAR metrics remain first-run untuned metrics and are not final benchmark claims.
- Coverage calibration remains weak:
  - TFT coverage_80_mean = 0.5833 (under-coverage vs 80% target interval).
  - DeepAR coverage_80_mean = 0.0000 (severe under-coverage / interval collapse behavior).
- Next major objective is calibration + focused search, not broad tuning.

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

Dry-run rolling-origin backtest execution:

```bash
uv run python -m lmp_forecaster.cli run-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24
```

Write rolling-origin backtest execution outputs:

```bash
uv run python -m lmp_forecaster.cli run-rolling-backtest \
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

## Rolling-origin evaluation execution

Module: `src/lmp_forecaster/eval/backtest_runner.py`

Includes:
- `BacktestRunConfig`
- `BacktestFoldResult`
- `RollingBacktestResult`
- `run_single_fold_backtest()`
- `run_rolling_backtest()`
- `aggregate_backtest_metrics()`
- `write_backtest_results()`
- `load_backtest_panel()`

Execution behavior:
- Dry-run prints validated folds/settings/expected paths and writes nothing.
- `--write` executes fold-by-fold training/evaluation and writes:
  - forecasts parquet,
  - per-fold metrics CSV,
  - aggregate metrics CSV,
  - summary JSON/Markdown,
  - artifact manifest JSON.

Coverage interpretation is explicit in reports:
- not near 80% target -> under-coverage,
- high above target -> over-coverage,
- only near target -> roughly calibrated.

Current real AEP run evidence (3 folds, horizon 24):
- TFT coverage_80_mean: 0.5833 (under-coverage)
- DeepAR coverage_80_mean: 0.0000 (under-coverage/interval collapse)


## Rolling-origin calibration diagnostics + focused search design

Step 9 adds calibration diagnostics and search-design-only outputs (no large tuning yet).

Diagnostics commands:

```bash
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP --write
```

Search design commands:

```bash
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP --write
```

Calibration evidence from rolling outputs:
- TFT remains under-covered relative to 80% target coverage.
- DeepAR remains severely under-covered with interval collapse warning behavior.

Search design intentionally limits cost and scope:
- first pass trial budget: 12,
- second pass cap: 30,
- no full broad Optuna run in this step,
- promotion gate requires coverage recovery plus bounded MAE regression.

## Artifact locations and ignore policy

Generated outputs remain local and gitignored:
- backtest outputs: `data/cache/backtests/`
- reports: `data/cache/reports/`
- baseline artifacts: `artifacts/baselines/`
- backtest artifacts: `artifacts/backtests/`
- tracking runs: `mlruns/`
- tracking scratch artifacts: `.mlflow_artifacts/`

## Interpretation note

Current metrics indicate:
- TFT currently outperforms DeepAR on MAE/RMSE for this untuned run.
- Both models still require calibration/tuning.
- DeepAR under-coverage should be treated as a model/config issue to fix, not as a final performance conclusion.
