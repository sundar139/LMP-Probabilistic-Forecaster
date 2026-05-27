"""Calibration diagnostics for rolling-origin probabilistic forecasts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lmp_forecaster.config.paths import get_project_paths


@dataclass(frozen=True)
class CalibrationDiagnosticConfig:
    target_coverage: float = 0.80
    tolerance: float = 0.05
    lower_quantile: float = 0.10
    median_quantile: float = 0.50
    upper_quantile: float = 0.90
    collapse_coverage_threshold: float = 0.05

    def validate(self) -> None:
        if not (0.0 < self.target_coverage < 1.0):
            raise ValueError("target_coverage must be in (0, 1)")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        if not (0.0 < self.lower_quantile < self.median_quantile < self.upper_quantile < 1.0):
            raise ValueError("Expected lower_quantile < median_quantile < upper_quantile in (0, 1)")
        if not (0.0 <= self.collapse_coverage_threshold < 1.0):
            raise ValueError("collapse_coverage_threshold must be in [0, 1)")


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (get_project_paths().root / path).resolve()


def _latest_file(pattern: str, root: Path) -> Path | None:
    files = list(root.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def load_calibration_config(config_path: Path | None = None) -> CalibrationDiagnosticConfig:
    root = get_project_paths().root
    path = config_path or (root / "conf" / "calibration.yaml")
    if not path.exists():
        cfg = CalibrationDiagnosticConfig()
        cfg.validate()
        return cfg

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("calibration config must parse to mapping")
    section = payload.get("calibration", payload)
    if not isinstance(section, dict):
        raise ValueError("calibration config section must be mapping")

    cfg = CalibrationDiagnosticConfig(
        target_coverage=float(section.get("target_coverage", 0.80)),
        tolerance=float(section.get("tolerance", 0.05)),
        lower_quantile=float(section.get("lower_quantile", 0.10)),
        median_quantile=float(section.get("median_quantile", 0.50)),
        upper_quantile=float(section.get("upper_quantile", 0.90)),
        collapse_coverage_threshold=float(section.get("collapse_coverage_threshold", 0.05)),
    )
    cfg.validate()
    return cfg


def discover_latest_backtest_outputs(
    zone: str = "AEP",
    *,
    backtest_root: Path = Path("data/cache/backtests"),
) -> dict[str, Path]:
    root = _resolve_path(backtest_root)
    if not root.exists():
        return {}

    zone_slug = zone.lower()
    forecasts = _latest_file(f"{zone_slug}_rolling_backtest_forecasts_*.parquet", root)
    fold_metrics = _latest_file(f"{zone_slug}_rolling_backtest_fold_metrics_*.csv", root)
    aggregate_metrics = _latest_file(f"{zone_slug}_rolling_backtest_aggregate_metrics_*.csv", root)

    out: dict[str, Path] = {}
    if forecasts is not None:
        out["forecasts"] = forecasts
    if fold_metrics is not None:
        out["fold_metrics"] = fold_metrics
    if aggregate_metrics is not None:
        out["aggregate_metrics"] = aggregate_metrics
    return out


def _require_forecast_columns(frame: pd.DataFrame) -> None:
    required = {
        "unique_id",
        "ds",
        "y",
        "p10",
        "p50",
        "p90",
        "model",
        "zone",
        "data_source_label",
        "fold_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Forecast frame missing required columns: {missing}")


def _prepare_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    _require_forecast_columns(forecasts)
    frame = forecasts.copy()

    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce", utc=False)
    if frame["ds"].isna().any():
        raise ValueError("Forecast ds contains invalid datetime values")

    for col in ["y", "p10", "p50", "p90"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            raise ValueError(f"Forecast column contains non-numeric values: {col}")

    frame["model"] = frame["model"].astype(str)
    frame = frame.sort_values(["model", "fold_id", "ds"]).reset_index(drop=True)
    frame["horizon_hour"] = frame.groupby(["model", "fold_id"], dropna=False).cumcount() + 1
    frame["inside_80"] = ((frame["y"] >= frame["p10"]) & (frame["y"] <= frame["p90"])).astype(float)
    frame["interval_width"] = frame["p90"] - frame["p10"]
    frame["residual_abs"] = (frame["y"] - frame["p50"]).abs()
    frame["median_residual"] = frame["p50"] - frame["y"]
    crossing_mask = (frame["p10"] > frame["p50"]) | (frame["p50"] > frame["p90"])
    frame["crossing"] = crossing_mask.astype(float)
    frame["p10_gt_p50"] = (frame["p10"] > frame["p50"]).astype(float)
    frame["p50_gt_p90"] = (frame["p50"] > frame["p90"]).astype(float)
    return frame


def compute_interval_coverage_by_horizon(forecasts: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_forecasts(forecasts)
    grouped = frame.groupby(["model", "horizon_hour"], dropna=False)
    return (
        grouped.agg(
            coverage_80=("inside_80", "mean"),
            row_count=("inside_80", "size"),
        )
        .reset_index()
        .sort_values(["model", "horizon_hour"])
        .reset_index(drop=True)
    )


def compute_interval_coverage_by_fold(forecasts: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_forecasts(forecasts)
    grouped = frame.groupby(["model", "fold_id", "zone", "data_source_label"], dropna=False)
    return (
        grouped.agg(
            coverage_80=("inside_80", "mean"),
            row_count=("inside_80", "size"),
            interval_width_mean=("interval_width", "mean"),
        )
        .reset_index()
        .sort_values(["model", "fold_id"])
        .reset_index(drop=True)
    )


def compute_interval_width_by_horizon(forecasts: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_forecasts(forecasts)
    grouped = frame.groupby(["model", "horizon_hour"], dropna=False)
    return (
        grouped.agg(
            interval_width_mean=("interval_width", "mean"),
            row_count=("interval_width", "size"),
        )
        .reset_index()
        .sort_values(["model", "horizon_hour"])
        .reset_index(drop=True)
    )


def compute_quantile_crossing_rate(forecasts: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_forecasts(forecasts)
    grouped = frame.groupby("model", dropna=False)
    return (
        grouped.agg(
            crossing_rate=("crossing", "mean"),
            p10_gt_p50_rate=("p10_gt_p50", "mean"),
            p50_gt_p90_rate=("p50_gt_p90", "mean"),
        )
        .reset_index()
        .sort_values("model")
        .reset_index(drop=True)
    )


def _pinball_loss(y_true: pd.Series, y_pred: pd.Series, q: float) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def compute_pinball_by_quantile(
    forecasts: pd.DataFrame,
    *,
    lower_quantile: float,
    median_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    frame = _prepare_forecasts(forecasts)
    rows: list[dict[str, Any]] = []

    for model, group in frame.groupby("model", dropna=False):
        pin10 = _pinball_loss(group["y"], group["p10"], lower_quantile)
        pin50 = _pinball_loss(group["y"], group["p50"], median_quantile)
        pin90 = _pinball_loss(group["y"], group["p90"], upper_quantile)
        rows.append(
            {
                "model": str(model),
                "pinball_p10": pin10,
                "pinball_p50": pin50,
                "pinball_p90": pin90,
                "mean_pinball_loss": float(np.mean([pin10, pin50, pin90])),
            }
        )

    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def compute_median_bias(forecasts: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_forecasts(forecasts)
    grouped = frame.groupby("model", dropna=False)
    return (
        grouped.agg(
            median_bias_mean=("median_residual", "mean"),
            median_bias_abs_mean=("median_residual", lambda s: float(s.abs().mean())),
            row_count=("median_residual", "size"),
        )
        .reset_index()
        .sort_values("model")
        .reset_index(drop=True)
    )


def classify_calibration_status(
    summary_by_model: pd.DataFrame,
    cfg: CalibrationDiagnosticConfig,
) -> pd.DataFrame:
    required = {"model", "coverage_80", "interval_width_mean"}
    missing = sorted(required.difference(summary_by_model.columns))
    if missing:
        raise ValueError(f"summary_by_model missing required fields: {missing}")

    frame = summary_by_model.copy().reset_index(drop=True)
    lower = cfg.target_coverage - cfg.tolerance
    upper = cfg.target_coverage + cfg.tolerance
    median_width = float(pd.to_numeric(frame["interval_width_mean"], errors="coerce").median())

    statuses: list[str] = []
    notes: list[str] = []
    collapse_flags: list[bool] = []

    for _, row in frame.iterrows():
        coverage = float(row["coverage_80"])
        width = float(row["interval_width_mean"])

        if coverage < lower:
            status = "under-coverage"
        elif coverage > upper:
            status = "over-coverage"
        else:
            status = "near target"

        narrow_interval = median_width > 0 and width <= (0.75 * median_width)
        collapse_warning = coverage <= cfg.collapse_coverage_threshold and narrow_interval

        if collapse_warning:
            note = "interval collapse warning"
            status = "under-coverage"
        else:
            note = status

        statuses.append(status)
        notes.append(note)
        collapse_flags.append(collapse_warning)

    frame["calibration_status"] = statuses
    frame["classification_note"] = notes
    frame["interval_collapse_warning"] = collapse_flags
    return frame


def summarize_calibration_diagnostics(
    forecasts: pd.DataFrame,
    *,
    cfg: CalibrationDiagnosticConfig,
    zone: str,
) -> dict[str, Any]:
    cfg.validate()
    frame = _prepare_forecasts(forecasts)

    coverage_by_fold = compute_interval_coverage_by_fold(frame)
    coverage_by_horizon = compute_interval_coverage_by_horizon(frame)
    width_by_horizon = compute_interval_width_by_horizon(frame)
    crossing = compute_quantile_crossing_rate(frame)
    pinball = compute_pinball_by_quantile(
        frame,
        lower_quantile=cfg.lower_quantile,
        median_quantile=cfg.median_quantile,
        upper_quantile=cfg.upper_quantile,
    )
    median_bias = compute_median_bias(frame)

    residual_mae_by_horizon = (
        frame.groupby(["model", "horizon_hour"], dropna=False)
        .agg(residual_mae=("residual_abs", "mean"), row_count=("residual_abs", "size"))
        .reset_index()
        .sort_values(["model", "horizon_hour"])
        .reset_index(drop=True)
    )

    model_base = (
        frame.groupby(["model", "zone", "data_source_label"], dropna=False)
        .agg(
            row_count=("y", "size"),
            coverage_80=("inside_80", "mean"),
            interval_width_mean=("interval_width", "mean"),
        )
        .reset_index()
    )

    model_summary = (
        model_base.merge(crossing, on="model", how="left")
        .merge(pinball, on="model", how="left")
        .merge(
            median_bias[["model", "median_bias_mean", "median_bias_abs_mean"]],
            on="model",
            how="left",
        )
    )
    model_summary = classify_calibration_status(model_summary, cfg)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "zone": zone,
        "rows_analyzed": int(len(frame)),
        "target_coverage": cfg.target_coverage,
        "tolerance": cfg.tolerance,
        "models": sorted([str(v) for v in frame["model"].astype(str).unique()]),
        "model_summary": model_summary.sort_values("model").to_dict(orient="records"),
        "coverage_by_fold": coverage_by_fold.to_dict(orient="records"),
        "coverage_by_horizon": coverage_by_horizon.to_dict(orient="records"),
        "interval_width_by_horizon": width_by_horizon.to_dict(orient="records"),
        "residual_mae_by_horizon": residual_mae_by_horizon.to_dict(orient="records"),
        "quantile_crossing": crossing.to_dict(orient="records"),
        "pinball_by_quantile": pinball.to_dict(orient="records"),
        "median_bias": median_bias.to_dict(orient="records"),
    }


def write_calibration_report(
    summary: dict[str, Any],
    *,
    zone: str,
    report_root: Path = Path("data/cache/reports"),
) -> tuple[Path, Path]:
    root = _resolve_path(report_root)
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    zone_slug = zone.lower()
    json_path = root / f"{zone_slug}_calibration_diagnostics_{stamp}.json"
    md_path = root / f"{zone_slug}_calibration_diagnostics_{stamp}.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_df = pd.DataFrame(summary.get("model_summary", []))
    fold_df = pd.DataFrame(summary.get("coverage_by_fold", []))

    lines = [
        f"# Calibration diagnostics — {zone}",
        "",
        f"Generated at: {summary.get('generated_at')}",
        f"Rows analyzed: {summary.get('rows_analyzed')}",
        f"Target coverage: {summary.get('target_coverage')}",
        f"Tolerance: {summary.get('tolerance')}",
        "",
        "## Model summary",
        model_df.to_markdown(index=False) if not model_df.empty else "(no model summary)",
        "",
        "## Coverage by fold",
        fold_df.to_markdown(index=False) if not fold_df.empty else "(no fold coverage)",
        "",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
