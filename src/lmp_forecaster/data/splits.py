"""Time-based split utilities for single-series panels."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplitConfig:
    """Configuration for deterministic chronological splits."""

    val_size: int = 72
    test_size: int = 72


def split_single_series_panel(
    panel: pd.DataFrame,
    config: TimeSplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a single-series panel into train/val/test by timestamp order."""
    for col in ["unique_id", "ds", "y"]:
        if col not in panel.columns:
            raise ValueError(f"Missing required column: {col}")

    if panel["unique_id"].astype(str).str.upper().nunique() != 1:
        raise ValueError("This step supports one unique_id only.")

    ordered = panel.sort_values("ds").reset_index(drop=True)
    n = len(ordered)

    required = config.val_size + config.test_size + 1
    if n < required:
        raise ValueError(
            f"Insufficient history for splits. Need at least {required} rows, got {n}."
        )

    train_end = n - (config.val_size + config.test_size)
    val_end = n - config.test_size

    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()

    validate_time_splits(train_df, val_df, test_df, config=config)
    return train_df, val_df, test_df


def validate_time_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    config: TimeSplitConfig,
) -> None:
    """Validate split chronology and non-overlap constraints."""
    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if frame.empty:
            raise ValueError(f"{name} split is empty.")
        if frame["ds"].isna().any():
            raise ValueError(f"{name} split contains null ds values.")

    train_ds = set(train_df["ds"].tolist())
    val_ds = set(val_df["ds"].tolist())
    test_ds = set(test_df["ds"].tolist())

    if train_ds & val_ds or train_ds & test_ds or val_ds & test_ds:
        raise ValueError("Time splits overlap; expected disjoint timestamps.")

    if not (train_df["ds"].max() < val_df["ds"].min() < test_df["ds"].min()):
        raise ValueError("Split chronology invalid; expected train < val < test in time.")

    if len(val_df) != config.val_size:
        raise ValueError(
            f"Validation split size mismatch. Expected {config.val_size}, got {len(val_df)}."
        )
    if len(test_df) != config.test_size:
        raise ValueError(
            f"Test split size mismatch. Expected {config.test_size}, got {len(test_df)}."
        )
