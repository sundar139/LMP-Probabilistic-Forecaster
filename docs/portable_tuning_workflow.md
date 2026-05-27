# Portable tuning execution and validation workflow

## Purpose

This workflow supports portable external tuning execution while keeping local validation authoritative.

It is designed for this local machine constraint envelope:
- 8GB VRAM
- 16GB RAM
- around 100GB disk

Full heavier tuning is not run locally because previous heavy runs were unstable and local resources are intentionally guarded.

## Resource profiles

Defined in `conf/tuning.yaml`:
- `local_safe`
  - local bounded smoke/planning profile
  - `max_trials=2`, `folds=1`, `max_steps_cap=3`, `batch_size=4`, `num_workers=0`
- `cloud_16gb`
  - external execution profile for 16GB VRAM-class hardware
  - `max_trials=12`, `folds=2`, `max_steps_cap=50`, `batch_size=8`
- `cloud_24gb`
  - external execution profile for 24GB+ VRAM-class hardware
  - `max_trials=30`, `folds=3`, `max_steps_cap=100`, `batch_size=16`

## Export a tuning package

Dry-run export (plan only; writes nothing):

```bash
uv run python -m lmp_forecaster.cli export-tuning-package \
  --zone AEP \
  --resource-profile cloud_16gb \
  --models TFT,DeepAR \
  --max-trials 12 \
  --folds 2
```

Alternate dry-run profile:

```bash
uv run python -m lmp_forecaster.cli export-tuning-package \
  --zone AEP \
  --resource-profile cloud_24gb \
  --models TFT,DeepAR \
  --max-trials 30 \
  --folds 3
```

Write manifest package:

```bash
uv run python -m lmp_forecaster.cli export-tuning-package \
  --zone AEP \
  --resource-profile cloud_16gb \
  --models TFT,DeepAR \
  --max-trials 12 \
  --folds 2 \
  --write
```

Write mode creates package manifests under ignored outputs:
- `data/cache/tuning_packages/<zone>_tuning_package_<timestamp>.json`
- `data/cache/tuning_packages/<zone>_tuning_package_<timestamp>.md`

Package manifest includes:
- commit hash and branch,
- uv command plan for external runner,
- selected resource profile and command plan,
- promotion gates,
- required repo/config references,
- import command template.

Package manifest excludes `.env` and API key secrets.

## Run tuning externally

Use manifest command plan on external runner:
1. sync environment and run quality checks,
2. run focused tuning with the package profile,
3. preserve ranked candidate CSV and summary outputs,
4. transfer only needed outputs back for local import.

Do not commit private data, credentials, or external runtime artifacts.

## Import ranked results locally

Dry-run import validation (writes nothing):

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

Optional explicit paths:
- `--summary-path <summary_json>`
- `--baseline-metrics-path <aggregate_metrics_csv>`

Write mode creates reports under ignored outputs:
- `data/cache/reports/<zone>_import_validation_<timestamp>.json`
- `data/cache/reports/<zone>_import_validation_<timestamp>.md`

## Local promotion recompute rule

Imported promotion labels are not trusted blindly.

Local import always recomputes promotion decisions from:
- imported candidate metrics,
- local baseline metrics,
- local promotion gate settings (`coverage_min`, `coverage_max`, `mae_regression_limit`, collapse/crossing rules).

Mismatch between imported label and local recompute status is detected and reported.

## If no candidate is promoted

If import result is `no_promotion`:
1. keep current baseline active,
2. run another external package with `cloud_16gb` or `cloud_24gb`,
3. import updated ranked results,
4. run 3-fold rolling backtest for the best candidate only,
5. promote only if rolling validation and local recompute gates pass.

## Reproduction command set

```bash
uv run python -m lmp_forecaster.cli export-tuning-package --zone AEP --resource-profile cloud_16gb --models TFT,DeepAR --max-trials 12 --folds 2
uv run python -m lmp_forecaster.cli export-tuning-package --zone AEP --resource-profile cloud_24gb --models TFT,DeepAR --max-trials 30 --folds 3
uv run python -m lmp_forecaster.cli export-tuning-package --zone AEP --resource-profile cloud_16gb --models TFT,DeepAR --max-trials 12 --folds 2 --write
uv run python -m lmp_forecaster.cli import-tuning-results --zone AEP --ranked-candidates-path <ranked_candidates_csv>
uv run python -m lmp_forecaster.cli import-tuning-results --zone AEP --ranked-candidates-path <ranked_candidates_csv> --write
```

## Recommended next action

Run `cloud_16gb` or `cloud_24gb` package externally, import ranked results locally, then rerun 3-fold rolling backtest for the single best imported candidate.
