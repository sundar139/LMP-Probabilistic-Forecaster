"""Rolling-origin backtest execution for real single-zone AEP panel data."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.eval.backtest import (
    BacktestConfig,
    BacktestFold,
    make_rolling_origin_folds,
    validate_backtest_folds,
)
from lmp_forecaster.eval.metrics import evaluate_probabilistic_forecast
from lmp_forecaster.models.baselines import (
    AcceleratorInfo,
    BaselineTrainingConfig,
    detect_accelerator,
    load_training_config,
    train_deepar_baseline,
    train_tft_baseline,
)
from lmp_forecaster.models.forecast_schema import validate_quantile_forecast
from lmp_forecaster.tracking.mlflow_utils import (
    TrackingConfig,
    configure_mlflow,
    log_artifact_paths,
    log_training_config,
    start_mlflow_run,
)

_SUPPORTED_MODELS = {"TFT", "DEEPAR"}


@dataclass(frozen=True)
class BacktestRunConfig:
    zone: str = "AEP"
    panel_path: Path = Path("data/processed/panel/single_zone/AEP_panel.parquet")
    folds: int = 3
    horizon_hours: int = 24
    min_train_hours: int = 2160
    input_size_hours: int = 168
    window_mode: str = "expanding"
    models: tuple[str, ...] = ("TFT", "DeepAR")
    max_steps: int = 30
    seed: int = 42
    skip_tft: bool = False
    skip_deepar: bool = False
    require_real_data: bool = True
    enable_tracking: bool = False
    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    output_root: Path = Path("data/cache/backtests")
    report_root: Path = Path("data/cache/reports")
    artifact_root: Path = Path("artifacts/backtests")

    def validate(self) -> None:
        if self.folds <= 0:
            raise ValueError("folds must be > 0")
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be > 0")
        if self.min_train_hours < self.input_size_hours:
            raise ValueError("min_train_hours must be >= input_size_hours")
        if self.window_mode not in {"expanding", "rolling"}:
            raise ValueError("window_mode must be expanding or rolling")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        normalized_models = [m.upper() for m in self.models]
        invalid = sorted({m for m in normalized_models if m not in _SUPPORTED_MODELS})
        if invalid:
            raise ValueError(f"Unsupported model(s): {invalid}")

        if not self.enabled_models:
            raise ValueError("At least one model must be enabled.")

    @property
    def enabled_models(self) -> list[str]:
        enabled: list[str] = []
        for model_name in self.models:
            key = model_name.upper()
            if key == "TFT" and self.skip_tft:
                continue
            if key == "DEEPAR" and self.skip_deepar:
                continue
            enabled.append("DeepAR" if key == "DEEPAR" else "TFT")
        return enabled


@dataclass(frozen=True)
class BacktestFoldResult:
    model: str
    fold_id: int
    forecast: pd.DataFrame
    metrics: dict[str, Any]


@dataclass
class RollingBacktestResult:
    config: BacktestRunConfig
    folds: list[BacktestFold]
    forecasts: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    accelerator: str
    device_name: str
    data_source_label: str
    tracking: dict[str, Any] | None = None
    output_paths: dict[str, str] | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return payload


def load_backtest_run_config(config_path: Path | None = None) -> BacktestRunConfig:
    root = get_project_paths().root
    path = config_path or (root / "conf" / "backtest.yaml")
    raw = _load_yaml(path)
    backtest = raw.get("backtest", raw)
    if not isinstance(backtest, dict):
        raise ValueError("conf/backtest.yaml must define a mapping")

    models_raw = backtest.get("models", ["TFT", "DeepAR"])
    if not isinstance(models_raw, list) or not all(isinstance(v, str) for v in models_raw):
        raise ValueError("backtest.models must be a list of strings")

    cfg = BacktestRunConfig(
        zone=str(backtest.get("zone", "AEP")).upper(),
        panel_path=Path(
            str(
                backtest.get(
                    "panel_path",
                    "data/processed/panel/single_zone/AEP_panel.parquet",
                )
            )
        ),
        folds=int(backtest.get("folds", 3)),
        horizon_hours=int(backtest.get("horizon_hours", 24)),
        min_train_hours=int(backtest.get("min_train_hours", 2160)),
        input_size_hours=int(backtest.get("input_size_hours", 168)),
        window_mode=str(backtest.get("window_mode", "expanding")),
        models=tuple(models_raw),
        max_steps=int(backtest.get("max_steps", 30)),
        seed=int(backtest.get("seed", 42)),
        enable_tracking=bool(backtest.get("enable_tracking", False)),
        tracking_uri=(
            str(backtest["tracking_uri"]) if backtest.get("tracking_uri") is not None else None
        ),
        experiment_name=(
            str(backtest["experiment_name"])
            if backtest.get("experiment_name") is not None
            else None
        ),
        output_root=Path(str(backtest.get("output_root", "data/cache/backtests"))),
        report_root=Path(str(backtest.get("report_root", "data/cache/reports"))),
        artifact_root=Path(str(backtest.get("artifact_root", "artifacts/backtests"))),
    )
    cfg.validate()
    return cfg


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (get_project_paths().root / path).resolve()


def planned_output_paths(cfg: BacktestRunConfig, stamp: str = "<timestamp>") -> dict[str, Path]:
    zone = cfg.zone.lower()
    backtest_root = _resolve_path(cfg.output_root)
    report_root = _resolve_path(cfg.report_root)
    artifact_root = _resolve_path(cfg.artifact_root)
    return {
        "forecasts": backtest_root / f"{zone}_rolling_backtest_forecasts_{stamp}.parquet",
        "fold_metrics": backtest_root / f"{zone}_rolling_backtest_fold_metrics_{stamp}.csv",
        "aggregate_metrics": (
            backtest_root / f"{zone}_rolling_backtest_aggregate_metrics_{stamp}.csv"
        ),
        "summary_json": report_root / f"{zone}_rolling_backtest_summary_{stamp}.json",
        "summary_markdown": report_root / f"{zone}_rolling_backtest_summary_{stamp}.md",
        "manifest": artifact_root / f"{zone}_rolling_backtest_manifest_{stamp}.json",
    }


def load_backtest_panel(
    panel_path: Path,
    *,
    zone: str,
    require_real_data: bool = True,
) -> tuple[pd.DataFrame, str]:
    resolved_path = _resolve_path(panel_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            "Real AEP panel is missing. Run build-single-zone-panel before rolling backtest."
        )

    panel = pd.read_parquet(resolved_path)
    required_cols = {"unique_id", "ds", "y"}
    missing_cols = sorted(required_cols.difference(panel.columns))
    if missing_cols:
        raise ValueError(f"Panel missing required columns: {missing_cols}")

    frame = panel.copy()
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce", utc=False)
    if frame["ds"].isna().any():
        raise ValueError("Panel ds contains invalid datetime values.")

    if "source_label" not in frame.columns:
        frame["source_label"] = "unknown"

    frame["unique_id"] = frame["unique_id"].astype(str)
    zone_upper = zone.upper()
    zone_panel = frame[frame["unique_id"].str.upper() == zone_upper].copy()
    if zone_panel.empty:
        raise ValueError(f"Panel does not contain rows for requested zone: {zone_upper}")

    zone_panel = zone_panel.sort_values("ds").reset_index(drop=True)
    source_label = str(zone_panel["source_label"].astype(str).mode().iloc[0]).lower()

    if require_real_data and source_label != "real":
        raise ValueError("Rolling backtest requires real panel data_source_label=real.")

    return zone_panel, source_label


def _build_training_config(run_cfg: BacktestRunConfig) -> BaselineTrainingConfig:
    base = load_training_config(run_cfg.zone, max_steps_smoke=run_cfg.max_steps)
    cfg = BaselineTrainingConfig(
        **{
            **base.__dict__,
            "zone": run_cfg.zone,
            "horizon": run_cfg.horizon_hours,
            "input_size": run_cfg.input_size_hours,
            "random_seed": run_cfg.seed,
            "max_steps_smoke": run_cfg.max_steps,
        }
    )
    return cfg


def _default_model_runner(
    model: str,
    train_df: pd.DataFrame,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_cfg: BaselineTrainingConfig,
    accel: AcceleratorInfo,
    data_source_label: str,
) -> pd.DataFrame:
    if model == "TFT":
        return train_tft_baseline(
            train_df,
            history_df,
            test_df,
            train_cfg,
            accel,
            data_source_label=data_source_label,
        )
    if model == "DeepAR":
        return train_deepar_baseline(
            train_df,
            history_df,
            test_df,
            train_cfg,
            accel,
            data_source_label=data_source_label,
        )
    raise ValueError(f"Unsupported model: {model}")


def run_single_fold_backtest(
    panel: pd.DataFrame,
    fold: BacktestFold,
    *,
    model: str,
    zone: str,
    data_source_label: str,
    train_cfg: BaselineTrainingConfig,
    accelerator: AcceleratorInfo,
    model_runner: Callable[
        [
            str,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            BaselineTrainingConfig,
            AcceleratorInfo,
            str,
        ],
        pd.DataFrame,
    ] = _default_model_runner,
) -> BacktestFoldResult:
    frame = panel.sort_values("ds").reset_index(drop=True)
    train_df = frame[frame["ds"] < fold.origin][["unique_id", "ds", "y"]].copy()
    test_df = frame[
        (frame["ds"] >= fold.test_start) & (frame["ds"] <= fold.test_end)
    ][["unique_id", "ds", "y"]].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Fold {fold.fold_id} has empty train or test slice.")
    if not bool((train_df["ds"] < fold.origin).all()):
        raise ValueError(f"Leakage detected in fold {fold.fold_id} training rows.")
    if not bool((test_df["ds"] >= fold.origin).all()):
        raise ValueError(f"Fold {fold.fold_id} test rows fall before origin.")

    history_df = train_df.copy()
    forecast = model_runner(
        model,
        train_df,
        history_df,
        test_df,
        train_cfg,
        accelerator,
        data_source_label,
    )
    validate_quantile_forecast(forecast)

    forecast_out = forecast.copy()
    forecast_out["fold_id"] = fold.fold_id

    quantiles_monotonic = (forecast_out["p10"] <= forecast_out["p50"]) & (
        forecast_out["p50"] <= forecast_out["p90"]
    )
    if not bool(quantiles_monotonic.all()):
        raise ValueError(f"Invalid quantile ordering in fold {fold.fold_id} for model {model}")

    metrics_raw = evaluate_probabilistic_forecast(
        forecast_out,
        model=model,
        zone=zone,
        data_source_label=data_source_label,
    )

    fold_metrics = {
        "model": model,
        "fold_id": fold.fold_id,
        "zone": zone,
        "row_count": int(len(forecast_out)),
        "test_start": str(fold.test_start),
        "test_end": str(fold.test_end),
        "MAE": float(metrics_raw["mae"]),
        "RMSE": float(metrics_raw["rmse"]),
        "pinball_p10": float(metrics_raw["pinball_p10"]),
        "pinball_p50": float(metrics_raw["pinball_p50"]),
        "pinball_p90": float(metrics_raw["pinball_p90"]),
        "mean_pinball_loss": float(metrics_raw["mean_pinball_loss"]),
        "coverage_80": float(metrics_raw["coverage_80"]),
        "interval_width_mean": float(metrics_raw["interval_width_mean"]),
        "data_source_label": data_source_label,
    }

    return BacktestFoldResult(
        model=model,
        fold_id=fold.fold_id,
        forecast=forecast_out,
        metrics=fold_metrics,
    )


def aggregate_backtest_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "fold_id",
        "zone",
        "row_count",
        "MAE",
        "RMSE",
        "mean_pinball_loss",
        "coverage_80",
        "interval_width_mean",
        "data_source_label",
    }
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"Fold metrics missing required fields: {missing}")

    rows: list[dict[str, Any]] = []
    grouped = fold_metrics.groupby(["model", "zone", "data_source_label"], dropna=False)
    for (model, zone, data_source_label), group in grouped:
        mae_series = pd.to_numeric(group["MAE"], errors="coerce")
        rmse_series = pd.to_numeric(group["RMSE"], errors="coerce")
        pinball_series = pd.to_numeric(group["mean_pinball_loss"], errors="coerce")
        coverage_series = pd.to_numeric(group["coverage_80"], errors="coerce")
        width_series = pd.to_numeric(group["interval_width_mean"], errors="coerce")

        rows.append(
            {
                "model": str(model),
                "zone": str(zone),
                "folds_completed": int(group["fold_id"].nunique()),
                "total_test_rows": int(pd.to_numeric(group["row_count"], errors="coerce").sum()),
                "MAE_mean": float(mae_series.mean()),
                "MAE_std": float(mae_series.std(ddof=0)),
                "RMSE_mean": float(rmse_series.mean()),
                "RMSE_std": float(rmse_series.std(ddof=0)),
                "mean_pinball_loss_mean": float(pinball_series.mean()),
                "mean_pinball_loss_std": float(pinball_series.std(ddof=0)),
                "coverage_80_mean": float(coverage_series.mean()),
                "coverage_80_std": float(coverage_series.std(ddof=0)),
                "interval_width_mean": float(width_series.mean()),
                "interval_width_std": float(width_series.std(ddof=0)),
                "best_fold_mae": float(mae_series.min()),
                "worst_fold_mae": float(mae_series.max()),
                "data_source_label": str(data_source_label),
            }
        )

    ordered_cols = [
        "model",
        "zone",
        "folds_completed",
        "total_test_rows",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "mean_pinball_loss_mean",
        "mean_pinball_loss_std",
        "coverage_80_mean",
        "coverage_80_std",
        "interval_width_mean",
        "interval_width_std",
        "best_fold_mae",
        "worst_fold_mae",
        "data_source_label",
    ]
    return pd.DataFrame(rows, columns=ordered_cols)


def run_rolling_backtest(
    run_cfg: BacktestRunConfig,
    *,
    model_runner: Callable[
        [
            str,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            BaselineTrainingConfig,
            AcceleratorInfo,
            str,
        ],
        pd.DataFrame,
    ] = _default_model_runner,
) -> RollingBacktestResult:
    run_cfg.validate()

    np.random.seed(run_cfg.seed)
    torch.manual_seed(run_cfg.seed)

    panel, data_source_label = load_backtest_panel(
        run_cfg.panel_path,
        zone=run_cfg.zone,
        require_real_data=run_cfg.require_real_data,
    )

    plan_cfg = BacktestConfig(
        zone=run_cfg.zone,
        horizon_hours=run_cfg.horizon_hours,
        folds=run_cfg.folds,
        min_train_hours=run_cfg.min_train_hours,
        window_mode=run_cfg.window_mode,
    )
    folds = make_rolling_origin_folds(panel, plan_cfg)
    validate_backtest_folds(folds)

    train_cfg = _build_training_config(run_cfg)
    accelerator = detect_accelerator(train_cfg.accelerator)

    fold_results: list[BacktestFoldResult] = []
    for fold in folds:
        for model_name in run_cfg.enabled_models:
            fold_results.append(
                run_single_fold_backtest(
                    panel,
                    fold,
                    model=model_name,
                    zone=run_cfg.zone,
                    data_source_label=data_source_label,
                    train_cfg=train_cfg,
                    accelerator=accelerator,
                    model_runner=model_runner,
                )
            )

    forecast_frames = [item.forecast for item in fold_results]
    metrics_rows = [item.metrics for item in fold_results]
    forecasts_df = pd.concat(forecast_frames, ignore_index=True)
    fold_metrics_df = pd.DataFrame(metrics_rows)
    aggregate_df = aggregate_backtest_metrics(fold_metrics_df)

    return RollingBacktestResult(
        config=run_cfg,
        folds=folds,
        forecasts=forecasts_df,
        fold_metrics=fold_metrics_df,
        aggregate_metrics=aggregate_df,
        accelerator=accelerator.kind,
        device_name=accelerator.device_name,
        data_source_label=data_source_label,
    )


def _coverage_interpretation(coverage: float) -> str:
    if coverage < 0.7:
        return "under-coverage"
    if coverage > 0.9:
        return "over-coverage"
    return "roughly calibrated"


def write_backtest_results(
    result: RollingBacktestResult,
    *,
    output_root: Path | None = None,
) -> dict[str, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cfg = result.config

    backtest_root = _resolve_path(output_root or cfg.output_root)
    report_root = _resolve_path(cfg.report_root)
    artifact_root = _resolve_path(cfg.artifact_root)

    backtest_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    zone = cfg.zone.lower()
    paths = {
        "forecasts": backtest_root / f"{zone}_rolling_backtest_forecasts_{stamp}.parquet",
        "fold_metrics": backtest_root / f"{zone}_rolling_backtest_fold_metrics_{stamp}.csv",
        "aggregate_metrics": (
            backtest_root / f"{zone}_rolling_backtest_aggregate_metrics_{stamp}.csv"
        ),
        "summary_json": report_root / f"{zone}_rolling_backtest_summary_{stamp}.json",
        "summary_markdown": report_root / f"{zone}_rolling_backtest_summary_{stamp}.md",
        "manifest": artifact_root / f"{zone}_rolling_backtest_manifest_{stamp}.json",
    }

    result.forecasts.to_parquet(paths["forecasts"], index=False)
    result.fold_metrics.to_csv(paths["fold_metrics"], index=False)
    result.aggregate_metrics.to_csv(paths["aggregate_metrics"], index=False)

    aggregate_rows = result.aggregate_metrics.to_dict(orient="records")
    coverage_notes = []
    for row in aggregate_rows:
        coverage = float(row["coverage_80_mean"])
        model = str(row["model"])
        note = _coverage_interpretation(coverage)
        coverage_notes.append(
            {
                "model": model,
                "coverage_80_mean": coverage,
                "interpretation": note,
            }
        )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "zone": cfg.zone,
        "models_attempted": cfg.enabled_models,
        "folds_requested": cfg.folds,
        "folds_completed": int(result.fold_metrics["fold_id"].nunique()),
        "horizon_hours": cfg.horizon_hours,
        "window_mode": cfg.window_mode,
        "data_source_label": result.data_source_label,
        "accelerator": result.accelerator,
        "device_name": result.device_name,
        "coverage_notes": coverage_notes,
        "fold_metrics_path": str(paths["fold_metrics"]),
        "aggregate_metrics_path": str(paths["aggregate_metrics"]),
        "forecasts_path": str(paths["forecasts"]),
        "aggregate_metrics": aggregate_rows,
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    aggregate_md = result.aggregate_metrics.to_markdown(index=False)
    fold_table = pd.DataFrame(
        [
            {
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "train_rows": fold.train_rows,
                "test_rows": fold.test_rows,
            }
            for fold in result.folds
        ]
    ).to_markdown(index=False)

    note_lines = [
        (
            f"- {item['model']}: coverage_80_mean={item['coverage_80_mean']:.4f} "
            f"-> {item['interpretation']}"
        )
        for item in coverage_notes
    ]
    md_lines = [
        f"# Rolling backtest summary — {cfg.zone}",
        "",
        f"Generated at: {summary['generated_at']}",
        f"Models attempted: {', '.join(cfg.enabled_models)}",
        f"Folds requested: {cfg.folds}",
        f"Folds completed: {summary['folds_completed']}",
        f"Horizon hours: {cfg.horizon_hours}",
        f"Window mode: {cfg.window_mode}",
        f"Data source label: {result.data_source_label}",
        f"Accelerator: {result.accelerator} ({result.device_name})",
        "",
        "## Fold structure",
        fold_table,
        "",
        "## Aggregate metrics",
        aggregate_md,
        "",
        "## Coverage interpretation",
        *note_lines,
    ]
    paths["summary_markdown"].write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "zone": cfg.zone,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result.output_paths = {k: str(v) for k, v in paths.items()}
    return paths


def log_backtest_tracking(
    result: RollingBacktestResult,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    cfg = result.config
    tracking_cfg = TrackingConfig(
        enabled=cfg.enable_tracking,
        experiment_name=cfg.experiment_name or "lmp_rolling_backtest",
        tracking_uri=cfg.tracking_uri or "file:./mlruns",
        run_name_prefix="rolling_backtest",
        log_artifacts=True,
        log_model_artifacts=False,
    )

    ctx = configure_mlflow(tracking_cfg)
    status: dict[str, Any] = {
        "enabled": ctx.enabled,
        "reason": ctx.reason,
        "tracking_uri": tracking_cfg.tracking_uri,
        "experiment_name": tracking_cfg.experiment_name,
        "run_id": None,
    }
    if not ctx.enabled:
        return status

    run_name = cfg.run_name or f"rolling_backtest_{cfg.zone.lower()}"
    with start_mlflow_run(ctx, run_name=run_name) as run:
        logged_params = log_training_config(
            run,
            {
                "zone": cfg.zone,
                "folds": cfg.folds,
                "horizon_hours": cfg.horizon_hours,
                "min_train_hours": cfg.min_train_hours,
                "window_mode": cfg.window_mode,
                "models": ",".join(cfg.enabled_models),
                "max_steps": cfg.max_steps,
                "data_source_label": result.data_source_label,
            },
        )

        import mlflow

        flat_metrics: dict[str, float] = {}
        for row in result.aggregate_metrics.to_dict(orient="records"):
            model = str(row["model"])
            flat_metrics[f"{model}_MAE_mean"] = float(row["MAE_mean"])
            flat_metrics[f"{model}_RMSE_mean"] = float(row["RMSE_mean"])
            flat_metrics[f"{model}_mean_pinball_loss_mean"] = float(
                row["mean_pinball_loss_mean"]
            )
            flat_metrics[f"{model}_coverage_80_mean"] = float(row["coverage_80_mean"])
        if flat_metrics:
            mlflow.log_metrics(flat_metrics)

        if tracking_cfg.log_artifacts:
            log_artifact_paths(
                run,
                {key: str(value) for key, value in artifact_paths.items()},
                artifact_file=f"rolling_backtest_artifacts_{cfg.zone}.txt",
            )

        status["logged_param_count"] = len(logged_params)
        run_info = getattr(run, "info", None)
        status["run_id"] = getattr(run_info, "run_id", None)

    return status
