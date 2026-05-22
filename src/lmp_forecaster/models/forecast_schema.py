"""Forecast schema normalization and validation utilities."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd


def infer_quantile_columns(columns: list[str]) -> dict[str, str]:
    """Infer p10/p50/p90 columns from common naming variants."""
    lowered = {c.lower(): c for c in columns}

    candidates = {
        "p10": ["p10", "q10", "0.1", "-lo-80", "lo-80"],
        "p50": ["p50", "q50", "0.5", "median"],
        "p90": ["p90", "q90", "0.9", "-hi-80", "hi-80"],
    }

    result: dict[str, str] = {}
    for target, opts in candidates.items():
        for opt in opts:
            for key, orig in lowered.items():
                if opt in key:
                    result[target] = orig
                    break
            if target in result:
                break

    missing = [k for k in ["p10", "p50", "p90"] if k not in result]
    if missing:
        raise ValueError(
            f"Could not infer quantile columns for {missing} from columns: {columns}"
        )
    return result


def normalize_neuralforecast_output(
    forecast: pd.DataFrame,
    *,
    model: str,
    actuals: pd.DataFrame | None = None,
    data_source_label: str | None = None,
    zone: str | None = None,
) -> pd.DataFrame:
    """Normalize forecast output to standard quantile schema."""
    required = ["unique_id", "ds"]
    for col in required:
        if col not in forecast.columns:
            raise ValueError(f"Forecast output missing required column: {col}")

    mapping = infer_quantile_columns(list(forecast.columns))
    out = forecast.copy()
    out = out.rename(
        columns={
            mapping["p10"]: "p10",
            mapping["p50"]: "p50",
            mapping["p90"]: "p90",
        }
    )

    out["ds"] = pd.to_datetime(out["ds"], errors="coerce", utc=False)
    if out["ds"].isna().any():
        raise ValueError("Forecast ds contains invalid datetime values.")

    for q in ["p10", "p50", "p90"]:
        out[q] = pd.to_numeric(out[q], errors="coerce")

    if actuals is not None:
        keep = actuals[["unique_id", "ds", "y"]].copy()
        keep["ds"] = pd.to_datetime(keep["ds"], errors="coerce", utc=False)
        out = out.merge(keep, on=["unique_id", "ds"], how="left")

    out["model"] = model
    out["generated_at"] = datetime.now(UTC)
    out["data_source_label"] = data_source_label if data_source_label is not None else pd.NA
    out["zone"] = zone if zone is not None else pd.NA

    ordered = [
        "unique_id",
        "ds",
        "y",
        "p10",
        "p50",
        "p90",
        "model",
        "generated_at",
        "data_source_label",
        "zone",
    ]
    for col in ordered:
        if col not in out.columns:
            out[col] = pd.NA

    return out[ordered]


def validate_quantile_forecast(frame: pd.DataFrame) -> None:
    """Validate quantile forecast schema and monotonic quantile ordering."""
    required = [
        "unique_id",
        "ds",
        "p10",
        "p50",
        "p90",
        "model",
        "generated_at",
        "data_source_label",
        "zone",
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required forecast columns: {missing}")

    if frame["unique_id"].isna().any():
        raise ValueError("Forecast unique_id contains null values.")

    if frame["model"].isna().any():
        raise ValueError("Forecast model contains null values.")

    if frame["generated_at"].isna().any():
        raise ValueError("Forecast generated_at contains null values.")

    if frame["data_source_label"].isna().any():
        raise ValueError("Forecast data_source_label contains null values.")

    if frame["zone"].isna().any():
        raise ValueError("Forecast zone contains null values.")

    if pd.to_datetime(frame["ds"], errors="coerce", utc=False).isna().any():
        raise ValueError("Forecast ds contains non-datetime values.")

    for col in ["p10", "p50", "p90"]:
        if pd.to_numeric(frame[col], errors="coerce").isna().any():
            raise ValueError(f"Forecast quantile column has NaN: {col}")

    bad = ~((frame["p10"] <= frame["p50"]) & (frame["p50"] <= frame["p90"]))
    if bad.any():
        raise ValueError("Invalid quantile ordering: expected p10 <= p50 <= p90.")
