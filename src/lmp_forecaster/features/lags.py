"""Lag and rolling feature engineering for LMP panels."""

from __future__ import annotations

import pandas as pd


def _validate_required(frame: pd.DataFrame) -> None:
    for col in ["unique_id", "ds", "y"]:
        if col not in frame.columns:
            raise ValueError(f"Missing required column: {col}")


def _sort_panel(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["unique_id", "ds"]).copy()
    out.reset_index(drop=True, inplace=True)
    return out


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe lag features grouped by series."""
    _validate_required(frame)
    out = _sort_panel(frame)
    grouped = out.groupby("unique_id", sort=False)["y"]

    for lag in [1, 2, 3, 24, 48, 168]:
        out[f"lmp_lag_{lag}"] = grouped.shift(lag)

    return out


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe rolling features from shifted target history."""
    _validate_required(frame)
    out = _sort_panel(frame)

    grouped = out.groupby("unique_id", sort=False)["y"]
    shifted = grouped.shift(1)

    roll24 = shifted.groupby(out["unique_id"], sort=False).rolling(24)
    roll168 = shifted.groupby(out["unique_id"], sort=False).rolling(168)

    out["lmp_rolling_mean_24"] = roll24.mean().reset_index(level=0, drop=True)
    out["lmp_rolling_std_24"] = roll24.std().reset_index(level=0, drop=True)
    out["lmp_rolling_min_24"] = roll24.min().reset_index(level=0, drop=True)
    out["lmp_rolling_max_24"] = roll24.max().reset_index(level=0, drop=True)

    out["lmp_rolling_mean_168"] = roll168.mean().reset_index(level=0, drop=True)
    out["lmp_rolling_std_168"] = roll168.std().reset_index(level=0, drop=True)

    return out


def validate_no_lag_leakage(frame: pd.DataFrame) -> None:
    """Verify lag-1 equals prior y within each series."""
    _validate_required(frame)
    if "lmp_lag_1" not in frame.columns:
        raise ValueError("Missing required lag feature: lmp_lag_1")

    out = _sort_panel(frame)
    for uid, grp in out.groupby("unique_id", sort=False):
        expected = grp["y"].shift(1)
        mask = expected.notna() & grp["lmp_lag_1"].notna()
        if not grp.loc[mask, "lmp_lag_1"].equals(expected.loc[mask]):
            raise ValueError(f"Lag leakage detected for unique_id={uid}")
