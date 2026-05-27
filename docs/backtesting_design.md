# Rolling-origin backtest design and execution (AEP)

## Goal

Run real rolling-origin evaluation on the real AEP panel for first baseline quality evidence, using fold planning plus fold-by-fold model execution.

## CLI

Dry-run (planning only, no training, no writes, no tracking):

```bash
uv run python -m lmp_forecaster.cli run-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24
```

Write execution (real fold training/evaluation + report output):

```bash
uv run python -m lmp_forecaster.cli run-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24 \
  --write
```

Optional controls:
- `--skip-tft` or `--skip-deepar`
- `--enable-tracking --tracking-uri <uri> --experiment-name <name>`
- `--max-steps <int>`

## Execution module

`src/lmp_forecaster/eval/backtest_runner.py` includes:
- `BacktestRunConfig`
- `BacktestFoldResult`
- `RollingBacktestResult`
- `load_backtest_panel()`
- `run_single_fold_backtest()`
- `run_rolling_backtest()`
- `aggregate_backtest_metrics()`
- `write_backtest_results()`
- `log_backtest_tracking()`

## Validation guarantees

Per run:
- panel existence validation with clear failure when missing,
- real-data requirement check (`data_source_label=real`) for CLI execution,
- rolling-origin fold planning/validation via `BacktestConfig` + `BacktestFold`.

Per fold:
- training rows strictly before fold origin,
- test rows only in fold test window,
- no leakage/overlap via fold validation,
- quantile ordering validation (`p10 <= p50 <= p90`),
- required forecast schema fields enforced.

## Latest real run snapshot

Configuration:
- zone: AEP
- models: TFT, DeepAR
- folds: 3
- horizon_hours: 24
- min_train_hours: 2160
- window_mode: expanding
- max_steps: 30
- accelerator selected: GPU (NVIDIA GeForce RTX 4070 Laptop GPU)

Fold structure:
- fold_id=1: train 2024-01-08 00:00:00-05:00 -> 2024-12-28 23:00:00-05:00, test 2024-12-29 00:00:00-05:00 -> 2024-12-29 23:00:00-05:00, train_rows=8544, test_rows=24
- fold_id=2: train 2024-01-08 00:00:00-05:00 -> 2024-12-29 23:00:00-05:00, test 2024-12-30 00:00:00-05:00 -> 2024-12-30 23:00:00-05:00, train_rows=8568, test_rows=24
- fold_id=3: train 2024-01-08 00:00:00-05:00 -> 2024-12-30 23:00:00-05:00, test 2024-12-31 00:00:00-05:00 -> 2024-12-31 23:00:00-05:00, train_rows=8592, test_rows=24

Aggregate metrics (first real untuned rolling run):

| model | folds_completed | total_test_rows | MAE_mean | RMSE_mean | mean_pinball_loss_mean | coverage_80_mean | interval_width_mean | data_source_label |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TFT | 3 | 72 | 5.5506 | 6.2262 | 1.8597 | 0.5833 | 13.4750 | real |
| DeepAR | 3 | 72 | 21.9064 | 22.4298 | 10.2479 | 0.0000 | 5.0474 | real |

Coverage interpretation:
- TFT coverage_80_mean=0.5833: under-coverage.
- DeepAR coverage_80_mean=0.0000: severe under-coverage/interval collapse behavior.

## Output paths

Generated outputs are written under ignored paths only:
- `data/cache/backtests/`
- `data/cache/reports/`
- `artifacts/backtests/`

## Limits and cautions

- single-zone only (AEP),
- untuned model settings,
- short fold count (3),
- no hyperparameter search yet,
- no multi-zone training yet.

These results are baseline evidence, not final benchmark claims.

## Calibration diagnostics and focused search design

Step 9 adds:
- calibration diagnostics module: `src/lmp_forecaster/eval/calibration.py`
- focused search design module: `src/lmp_forecaster/tuning/search_design.py`
- CLI commands:
  - `analyze-calibration`
  - `design-focused-search`

Diagnostics from Step 8 rolling outputs confirm:
- TFT: under-coverage (`coverage_80=0.5833`) with wider intervals than DeepAR but still below target.
- DeepAR: severe under-coverage (`coverage_80=0.0000`) and interval collapse warning.

Search design outputs intentionally avoid large tuning at this stage:
- first pass: 12 trials (small budget)
- second pass cap: 30
- primary metric: `mean_pinball_loss`
- secondary metric: `coverage_80`
- promotion gate:
  - coverage in `[0.70, 0.90]`
  - no collapse warning (DeepAR)
  - <= 15% MAE regression vs baseline

CLI examples:

```bash
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP
uv run python -m lmp_forecaster.cli analyze-calibration --zone AEP --write
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP
uv run python -m lmp_forecaster.cli design-focused-search --zone AEP --write
```

## Next step

Step 10: execute a small focused search pass (Optuna or constrained manual grid) using the generated search design and promotion gates from Step 9 diagnostics.
