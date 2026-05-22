# Rolling-origin backtest design scaffold

## Goal

Prepare a reliable fold-planning scaffold for rolling-origin evaluation before expensive full backtest training.

## CLI

Dry-run planning:

```bash
uv run python -m lmp_forecaster.cli plan-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24
```

Write planning reports:

```bash
uv run python -m lmp_forecaster.cli plan-rolling-backtest \
  --zone AEP \
  --panel-path data/processed/panel/single_zone/AEP_panel.parquet \
  --folds 3 \
  --horizon-hours 24 \
  --write
```

## Fold planning module

`src/lmp_forecaster/eval/backtest.py` provides:
- `BacktestConfig`
- `BacktestFold`
- `make_rolling_origin_folds()`
- `validate_backtest_folds()`
- `summarize_backtest_plan()`
- `write_backtest_plan()`

## Supported planning modes

- `expanding`
- `rolling`

## Validation guarantees

Each fold enforces:
- `train_end < test_start`
- no leakage at/after origin in training range
- non-overlapping test windows
- minimum train history before each fold
- timezone preservation when panel timestamps are timezone-aware

## Write behavior

- Dry-run prints plan summary and writes nothing.
- `--write` writes JSON/Markdown under ignored report paths:
  - `data/cache/reports/backtest_plan_AEP_<timestamp>.json`
  - `data/cache/reports/backtest_plan_AEP_<timestamp>.md`

## Scope limits

This scaffold is planning-only for this step.
It does not train models across all folds yet.
