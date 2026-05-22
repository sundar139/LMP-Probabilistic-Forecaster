"""Evaluation metrics for probabilistic forecasts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, q: float) -> float:
    errors = y_true - y_pred
    loss = np.maximum(q * errors, (q - 1.0) * errors)
    return float(np.mean(loss))


def interval_coverage(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside.astype(float)))


def interval_width(lower: pd.Series, upper: pd.Series) -> float:
    return float(np.mean(upper - lower))


def evaluate_probabilistic_forecast(
    forecast: pd.DataFrame,
    *,
    model: str,
    zone: str,
    data_source_label: str,
) -> dict[str, Any]:
    """Evaluate quantile forecast against actual target values."""
    required = ["y", "p10", "p50", "p90"]
    missing = [c for c in required if c not in forecast.columns]
    if missing:
        raise ValueError(f"Forecast missing required columns for evaluation: {missing}")

    if forecast["y"].isna().any():
        raise ValueError("Forecast actual target y has missing values; cannot evaluate.")

    y = pd.to_numeric(forecast["y"], errors="coerce")
    p10 = pd.to_numeric(forecast["p10"], errors="coerce")
    p50 = pd.to_numeric(forecast["p50"], errors="coerce")
    p90 = pd.to_numeric(forecast["p90"], errors="coerce")

    if y.isna().any() or p10.isna().any() or p50.isna().any() or p90.isna().any():
        raise ValueError("Evaluation inputs contain non-numeric values.")

    pin10 = pinball_loss(y, p10, 0.1)
    pin50 = pinball_loss(y, p50, 0.5)
    pin90 = pinball_loss(y, p90, 0.9)

    return {
        "model": model,
        "zone": zone,
        "row_count": int(len(forecast)),
        "mae": mae(y, p50),
        "rmse": rmse(y, p50),
        "pinball_p10": pin10,
        "pinball_p50": pin50,
        "pinball_p90": pin90,
        "mean_pinball_loss": float(np.mean([pin10, pin50, pin90])),
        "coverage_80": interval_coverage(y, p10, p90),
        "interval_width_mean": interval_width(p10, p90),
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source_label": data_source_label,
    }
