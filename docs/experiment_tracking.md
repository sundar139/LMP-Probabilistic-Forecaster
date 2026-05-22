# Experiment tracking (MLflow)

## Objective

Provide optional, production-style experiment tracking for baseline runs without making normal CLI usage fragile.

## Defaults

Tracking is disabled by default.

Configuration file: `conf/tracking.yaml`

Default values:
- `enabled: false`
- `experiment_name: lmp_probabilistic_forecaster`
- `tracking_uri: file:./mlruns`
- `run_name_prefix: baseline`
- `log_artifacts: true`
- `log_model_artifacts: false`

`mlruns/` remains gitignored.

## CLI usage

Enable tracking for a write run:

```bash
uv run python -m lmp_forecaster.cli train-single-zone-baselines \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --enable-tracking \
  --experiment-name lmp_probabilistic_forecaster_smoke \
  --tracking-uri file:./mlruns \
  --run-name aep_smoke_tracking \
  --write
```

Dry-run never starts training or tracking.

## Logged information

When enabled and available, tracking logs:
- training config summary (zone, horizon, input_size, quantiles, interval_level, seed, split sizes, accelerator kind)
- model metrics (e.g. `TFT_mae`, `TFT_rmse`, `TFT_coverage_80`, `DeepAR_mae`, `DeepAR_rmse`, `DeepAR_coverage_80`)
- artifact path references as metadata (not raw private data payloads)

## Safety and redaction

Secret-like parameter keys are skipped from logging if key names contain markers such as:
- `key`, `token`, `secret`, `password`, `api`, `credential`

Training remains functional if MLflow is unavailable at runtime.
In that case, tracking status is reported as disabled with reason.

## Scope limits in this step

- This step introduces optional local tracking support.
- It does not introduce remote tracking server dependency.
- It does not change model selection/tuning policy.
