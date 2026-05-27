# Focused tuning execution guide (resource-safe local closeout)

## Scope

This guide documents the bounded focused tuning workflow for local constrained hardware.

Local resource baseline used for this step:
- GPU VRAM: 8GB
- System RAM: 16GB
- Disk headroom: around 100GB

A previous write-mode tuning attempt froze/timed out under heavier settings. Because of that, full first-pass search remains deferred on this machine.

## What this workflow is for

- Validate the focused tuning system end-to-end with strict resource caps.
- Produce honest smoke tuning metrics and a promotion summary.
- Prevent accidental heavy execution with local-safe guards.

## What this workflow is not for

- It is not final model selection.
- It is not multi-fold/multi-zone production validation.
- It does not replace full tuning on stronger hardware.

## Config profile

`conf/tuning.yaml` defines a local profile:

- `resource_profile.name: local_8gb_vram_16gb_ram`
- `max_trials_safe: 2`
- `folds_safe: 1`
- `max_steps_cap_safe: 3`
- `batch_size_safe: 4`
- `num_workers: 0`
- `cleanup_after_trial: true`
- `full_search_deferred: true`

The larger first pass remains documented but deferred:
- `max_trials_first_pass: 12`
- `folds_for_full_first_pass: 2`
- `status: deferred_on_local_machine`

## Commands

Dry-run (plan only, writes nothing):

```bash
uv run python -m lmp_forecaster.cli run-focused-tuning \
  --zone AEP \
  --resource-profile local_safe \
  --max-trials 2 \
  --folds 1 \
  --max-steps-cap 3 \
  --skip-deepar
```

Write-mode smoke run (bounded):

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

If local GPU behavior is unstable, force CPU:

```bash
uv run python -m lmp_forecaster.cli run-focused-tuning \
  --zone AEP \
  --resource-profile local_safe \
  --max-trials 1 \
  --folds 1 \
  --max-steps-cap 1 \
  --skip-deepar \
  --cpu-only \
  --write
```

## Safety behavior

- Local-safe mode refuses heavy values unless `--allow-heavy-run` is passed.
- `num_workers=0` in local-safe mode.
- Batch size defaults to local-safe minimum.
- Trial cleanup runs after each trial (artifacts + GC + CUDA cache clear where available).
- MLflow is optional and off by default (`--no-mlflow` available).
- No Optuna DB is created by default in local-safe mode.

## Promotion semantics under smoke scope

Promotion output is honest for limited evidence:
- `promoted`
- `rejected`
- `no_promotion`
- `smoke_candidate_requires_full_validation`
- `failed_resource_limited`

For 1-fold / 2-trial smoke runs, any promising candidate still requires multi-fold validation before final promotion.

## Output locations (ignored)

- `data/cache/tuning/`
- `data/cache/reports/`
- `artifacts/tuning/`

These paths remain git-ignored.

## Recommended next step

For meaningful promotion decisions:
1. Run a slightly larger focused search on cloud or stronger GPU hardware.
2. Re-run multi-fold rolling backtest on any candidate.
3. Only then consider final promotion.
