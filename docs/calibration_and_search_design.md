# Calibration diagnostics and focused search design (AEP)

## Why this step exists

Step 8 delivered real rolling-origin evidence but showed calibration gaps:
- TFT is materially better on error metrics but still under-covered (`coverage_80=0.5833`).
- DeepAR shows severe under-coverage (`coverage_80=0.0000`) with interval collapse warning behavior.

This step adds diagnostics + search design so Step 10 can execute a small targeted tuning pass safely.

## What was added

- `src/lmp_forecaster/eval/calibration.py`
  - coverage by model/fold/horizon
  - interval width by horizon
  - quantile crossing rate
  - pinball by quantile
  - median bias
  - calibration classification with collapse warning rule
- `src/lmp_forecaster/tuning/search_design.py`
  - model-specific focused search spaces for TFT and DeepAR
  - parameter rationale, expected effect, priority, and safe ranges
  - first-pass and second-pass trial budgets
  - stop and promotion criteria
- CLI commands:
  - `analyze-calibration`
  - `design-focused-search`

## Commands

```bash
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP --write
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP --write
```

## Current diagnostics summary (from Step 8 rolling outputs)

- TFT:
  - coverage_80: 0.5833 (under-coverage)
  - interval_width_mean: 13.4750
  - crossing_rate: 0.0000
  - median_bias: +3.9850
- DeepAR:
  - coverage_80: 0.0000 (under-coverage, collapse warning)
  - interval_width_mean: 5.0474
  - crossing_rate: 0.0000
  - median_bias: -21.9064

## Focused search design summary

- first pass: 12 trials
- second pass cap: 30 trials
- primary metric: `mean_pinball_loss`
- secondary metric: `coverage_80`
- promotion gate:
  - coverage in `[0.70, 0.90]`
  - no collapse warning (especially for DeepAR)
  - MAE regression no worse than 15% vs baseline

## What this step intentionally does not do

- does not run full broad Optuna search
- does not run large compute sweeps
- does not expand to multi-zone training

## Step 10 execution target

Run a small focused tuning pass (Optuna or constrained manual grid) directly from these search spaces and promotion gates, then compare promoted runs to Step 8 baseline metrics.
