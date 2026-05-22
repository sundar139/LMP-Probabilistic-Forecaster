# Baseline training

## Training objective

Train single-zone probabilistic baselines on the cleaned panel:
- TFT baseline
- DeepAR benchmark

Both produce P10/P50/P90 outputs for short-horizon evaluation.

## Data requirements

Input panel must include at least:
- `unique_id`
- `ds`
- `y`

For this step, one zone only (default `AEP`).

## Split policy

Chronological splits, no shuffle:
- train
- validation (default 72 hours)
- test (default 72 hours)

Constraints:
- no overlap
- `train_max_ds < val_min_ds < test_min_ds`

## TFT baseline

Implemented via `neuralforecast.models.TFT` with conservative smoke defaults:
- `h=24`
- `input_size=168`
- quantile loss for `[0.1, 0.5, 0.9]`

## DeepAR benchmark

Implemented via `neuralforecast.models.DeepAR` with conservative smoke defaults and the same horizon/input size.

## Quantile forecast schema

Normalized output columns:
- `unique_id`
- `ds`
- `y` (if actuals joined)
- `p10`
- `p50`
- `p90`
- `model`
- `generated_at`

Validation enforces `p10 <= p50 <= p90` and no quantile nulls.

## Metrics

Reported metrics:
- MAE
- RMSE
- pinball loss at p10/p50/p90
- mean pinball loss
- coverage_80 (`y` in [p10, p90])
- interval width mean (`p90 - p10`)

## Artifact layout

Local-only outputs:
- forecasts: `data/cache/forecasts/`
- reports/metrics: `data/cache/reports/`
- baseline artifacts: `artifacts/baselines/`

All are ignored by Git.

## Synthetic smoke-training policy

If panel is missing and synthetic fallback is enabled, the training path builds/uses synthetic panel data for smoke tests only.

Warning text is printed in CLI output. Synthetic metrics must not be presented as PJM performance.

## Known limitations

- Single-zone only for now.
- No rolling-origin backtests yet.
- No hyperparameter tuning yet.

## Next steps

- Fix PJM live endpoint/query format.
- Connect real cached AEP LMP data into panel and baseline training path.
- Add rolling-origin backtesting.
