# Rolling backtest results (AEP)

## Scope

This report captures the first real rolling-origin execution for AEP using untuned TFT and DeepAR baselines.

- zone: AEP
- folds requested: 3
- folds completed: 3
- horizon_hours: 24
- window_mode: expanding
- models attempted: TFT, DeepAR
- data_source_label: real

## Fold structure

| fold_id | train_start | train_end | test_start | test_end | train_rows | test_rows |
|---:|---|---|---|---|---:|---:|
| 1 | 2024-01-08 00:00:00-05:00 | 2024-12-28 23:00:00-05:00 | 2024-12-29 00:00:00-05:00 | 2024-12-29 23:00:00-05:00 | 8544 | 24 |
| 2 | 2024-01-08 00:00:00-05:00 | 2024-12-29 23:00:00-05:00 | 2024-12-30 00:00:00-05:00 | 2024-12-30 23:00:00-05:00 | 8568 | 24 |
| 3 | 2024-01-08 00:00:00-05:00 | 2024-12-30 23:00:00-05:00 | 2024-12-31 00:00:00-05:00 | 2024-12-31 23:00:00-05:00 | 8592 | 24 |

Leakage and overlap checks passed for planned folds.

## Aggregate metrics

| model | folds_completed | total_test_rows | MAE_mean | MAE_std | RMSE_mean | RMSE_std | mean_pinball_loss_mean | coverage_80_mean | interval_width_mean | data_source_label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TFT | 3 | 72 | 5.5506 | 2.0961 | 6.2262 | 1.9924 | 1.8597 | 0.5833 | 13.4750 | real |
| DeepAR | 3 | 72 | 21.9064 | 1.5873 | 22.4298 | 1.7163 | 10.2479 | 0.0000 | 5.0474 | real |

## Interpretation (honest)

- TFT is materially better than DeepAR on MAE/RMSE in this untuned rolling-origin run.
- TFT intervals are still under-calibrated for an 80% target (`coverage_80_mean=0.5833`), so this is under-coverage.
- DeepAR remains weak in this configuration, with near-collapsed uncertainty behavior (`coverage_80_mean=0.0000`), which is a model/config issue for next-step calibration/search.
- These are baseline diagnostics, not final benchmark claims.

## Limitations

- single-zone only (AEP),
- untuned hyperparameters,
- short fold count (3),
- no Optuna search yet,
- no multi-zone training yet.

## Suggested next step

Step 10: execute a small focused search pass (Optuna or constrained manual grid) using the generated search design, then compare promotion-gated candidates against Step 8 baselines.
