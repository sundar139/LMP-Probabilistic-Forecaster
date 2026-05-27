# Calibration diagnostics and focused search design (AEP)

## Why this step exists

Step 8 delivered real rolling-origin evidence but showed calibration gaps:
- TFT is materially better on error metrics but still under-covered (`coverage_80=0.5833`).
- DeepAR shows severe under-coverage (`coverage_80=0.0000`) with interval collapse warning behavior.

Step 9 added diagnostics + search design so focused tuning could run with bounded local resources.
Step 10 closeout executes only a hardware-safe smoke scope on this machine.

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
- `src/lmp_forecaster/tuning/tuning_runner.py`
  - local-safe profile enforcement
  - heavy-run guard for trial/fold/step limits
  - cleanup-after-trial hooks and runtime accounting
  - resource-limited failure handling (including CUDA OOM classification)
- CLI commands:
  - `analyze-calibration`
  - `design-focused-search`
  - `run-focused-tuning` with `--resource-profile local_safe` and `--allow-heavy-run`

## Commands

```bash
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP --write
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP --write
uv run python -m lmp_forecaster.cli run-focused-tuning --zone AEP --resource-profile local_safe --max-trials 2 --folds 1 --max-steps-cap 3 --skip-deepar
uv run python -m lmp_forecaster.cli run-focused-tuning --zone AEP --resource-profile local_safe --max-trials 2 --folds 1 --max-steps-cap 3 --skip-deepar --write
```

## Local hardware profile used for closeout

- 8GB VRAM
- 16GB RAM
- around 100GB disk

Because heavier write-mode tuning previously froze/timed out on this machine, full first-pass search is deferred here.

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

- first pass design: 12 trials
- folds for full first pass: 2
- local machine status: deferred_on_local_machine
- smoke scope executed locally: 2 trials, 1 fold, max_steps_cap 3
- primary metric: `mean_pinball_loss`
- secondary metric: `coverage_80`
- promotion gate:
  - coverage in `[0.70, 0.90]`
  - no collapse warning (especially for DeepAR)
  - MAE regression no worse than 15% vs baseline

## Smoke result interpretation

Smoke write-mode validates workflow safety and reproducibility only.
It does not provide enough evidence for final promotion.

For this local closeout run, promotion summary was `no_promotion` with coverage gate failures.

## What this step intentionally does not do

- does not run full broad Optuna tuning locally
- does not run large compute sweeps on constrained hardware
- does not expand to multi-zone training

## Recommended next step

Run slightly larger focused tuning on cloud/stronger GPU, then run multi-fold rolling backtest on any candidate before considering promotion.
