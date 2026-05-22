"""Rolling-origin backtest fold planning utilities (design scaffold only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lmp_forecaster.config.paths import get_project_paths


@dataclass(frozen=True)
class BacktestConfig:
    zone: str
    horizon_hours: int = 24
    folds: int = 3
    min_train_hours: int = 2160
    window_mode: str = "expanding"


@dataclass(frozen=True)
class BacktestFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    origin: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_rows: int
    test_rows: int
    horizon_hours: int
    leakage_check_passed: bool
    overlap_check_passed: bool


def _ensure_datetime_series(panel: pd.DataFrame) -> pd.Series:
    if "ds" not in panel.columns:
        raise ValueError("Panel missing required column: ds")
    ds = pd.to_datetime(panel["ds"], errors="coerce", utc=False)
    if ds.isna().any():
        raise ValueError("Panel ds contains non-datetime values")
    return ds


def make_rolling_origin_folds(panel: pd.DataFrame, cfg: BacktestConfig) -> list[BacktestFold]:
    if cfg.folds <= 0:
        raise ValueError("folds must be positive")
    if cfg.horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    if cfg.min_train_hours <= 0:
        raise ValueError("min_train_hours must be positive")
    if cfg.window_mode not in {"expanding", "rolling"}:
        raise ValueError("window_mode must be 'expanding' or 'rolling'")

    ds = _ensure_datetime_series(panel)
    ordered = panel.assign(ds=ds).sort_values("ds").reset_index(drop=True)
    n = len(ordered)

    required = cfg.min_train_hours + cfg.folds * cfg.horizon_hours
    if n < required:
        raise ValueError(
            f"Insufficient rows for backtest planning: rows={n}, required_at_least={required}"
        )

    folds: list[BacktestFold] = []
    test_block_total = cfg.folds * cfg.horizon_hours
    first_test_start_idx = n - test_block_total

    for i in range(cfg.folds):
        test_start_idx = first_test_start_idx + i * cfg.horizon_hours
        test_end_idx = test_start_idx + cfg.horizon_hours - 1
        train_end_idx = test_start_idx - 1

        if train_end_idx < 0:
            raise ValueError("No training rows available before fold test window")

        if cfg.window_mode == "expanding":
            train_start_idx = 0
        else:
            train_start_idx = max(0, train_end_idx - cfg.min_train_hours + 1)

        train_rows = train_end_idx - train_start_idx + 1
        if train_rows < cfg.min_train_hours:
            raise ValueError(
                f"Fold {i + 1} has insufficient training rows: {train_rows} < {cfg.min_train_hours}"
            )

        train_slice = ordered.iloc[train_start_idx : train_end_idx + 1]
        test_slice = ordered.iloc[test_start_idx : test_end_idx + 1]

        train_end = pd.Timestamp(train_slice["ds"].iloc[-1])
        test_start = pd.Timestamp(test_slice["ds"].iloc[0])
        origin = test_start

        leakage_ok = bool((train_slice["ds"] < origin).all()) and bool(train_end < test_start)

        overlap_ok = True
        if folds:
            prev = folds[-1]
            overlap_ok = prev.test_end < test_start

        fold = BacktestFold(
            fold_id=i + 1,
            train_start=pd.Timestamp(train_slice["ds"].iloc[0]),
            train_end=train_end,
            origin=origin,
            test_start=test_start,
            test_end=pd.Timestamp(test_slice["ds"].iloc[-1]),
            train_rows=train_rows,
            test_rows=len(test_slice),
            horizon_hours=cfg.horizon_hours,
            leakage_check_passed=leakage_ok,
            overlap_check_passed=overlap_ok,
        )
        folds.append(fold)

    return folds


def validate_backtest_folds(folds: list[BacktestFold]) -> None:
    if not folds:
        raise ValueError("No folds generated")

    for idx, fold in enumerate(folds):
        if not fold.train_end < fold.test_start:
            raise ValueError(f"Fold {fold.fold_id} invalid: train_end must be < test_start")
        if not fold.leakage_check_passed:
            raise ValueError(f"Fold {fold.fold_id} leakage validation failed")
        if not fold.overlap_check_passed:
            raise ValueError(f"Fold {fold.fold_id} overlap validation failed")
        if idx > 0 and not folds[idx - 1].test_end < fold.test_start:
            raise ValueError("Overlapping test windows detected")


def summarize_backtest_plan(
    panel: pd.DataFrame,
    cfg: BacktestConfig,
    folds: list[BacktestFold],
    output_path: Path,
) -> dict[str, Any]:
    ds = _ensure_datetime_series(panel)
    timezone = str(ds.dt.tz) if ds.dt.tz is not None else None

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "zone": cfg.zone,
        "window_mode": cfg.window_mode,
        "horizon_hours": cfg.horizon_hours,
        "folds_requested": cfg.folds,
        "panel_row_count": int(len(panel)),
        "panel_min_ds": str(ds.min()),
        "panel_max_ds": str(ds.max()),
        "timezone": timezone,
        "output_path": str(output_path),
        "folds": [
            {
                "fold_id": f.fold_id,
                "train_start": str(f.train_start),
                "train_end": str(f.train_end),
                "origin": str(f.origin),
                "test_start": str(f.test_start),
                "test_end": str(f.test_end),
                "train_rows": f.train_rows,
                "test_rows": f.test_rows,
                "horizon_hours": f.horizon_hours,
                "leakage_check_passed": f.leakage_check_passed,
                "overlap_check_passed": f.overlap_check_passed,
            }
            for f in folds
        ],
    }


def write_backtest_plan(
    summary: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    root = output_dir or (get_project_paths().root / "data" / "cache" / "reports")
    root.mkdir(parents=True, exist_ok=True)

    zone = str(summary.get("zone", "UNKNOWN"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = root / f"backtest_plan_{zone}_{stamp}.json"
    md_path = root / f"backtest_plan_{zone}_{stamp}.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Rolling-origin backtest plan — {zone}",
        "",
        f"Generated at: {summary.get('generated_at')}",
        f"Window mode: {summary.get('window_mode')}",
        f"Horizon hours: {summary.get('horizon_hours')}",
        f"Panel rows: {summary.get('panel_row_count')}",
        f"Panel range: {summary.get('panel_min_ds')} -> {summary.get('panel_max_ds')}",
        f"Timezone: {summary.get('timezone')}",
        "",
        "## Folds",
    ]

    for fold in summary.get("folds", []):
        row = fold if isinstance(fold, dict) else {}
        lines.extend(
            [
                f"- fold_id: {row.get('fold_id')}",
                f"  train_start: {row.get('train_start')}",
                f"  train_end: {row.get('train_end')}",
                f"  origin: {row.get('origin')}",
                f"  test_start: {row.get('test_start')}",
                f"  test_end: {row.get('test_end')}",
                f"  train_rows: {row.get('train_rows')}",
                f"  test_rows: {row.get('test_rows')}",
                f"  horizon_hours: {row.get('horizon_hours')}",
                f"  leakage_check_passed: {row.get('leakage_check_passed')}",
                f"  overlap_check_passed: {row.get('overlap_check_passed')}",
            ]
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
