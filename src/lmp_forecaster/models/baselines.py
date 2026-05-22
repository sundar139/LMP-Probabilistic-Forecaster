"""Single-zone baseline training utilities using NeuralForecast TFT and DeepAR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
from neuralforecast.models import TFT, DeepAR

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.build_panel import (
    PanelBuildConfig,
    build_single_zone_panel,
    validate_panel_schema,
)
from lmp_forecaster.data.splits import TimeSplitConfig, split_single_series_panel
from lmp_forecaster.eval.metrics import evaluate_probabilistic_forecast
from lmp_forecaster.models.forecast_schema import (
    normalize_neuralforecast_output,
    validate_quantile_forecast,
)


@dataclass(frozen=True)
class BaselineTrainingConfig:
    zone: str = "AEP"
    panel_path: Path | None = None
    allow_synthetic_panel: bool = False
    build_panel_if_missing: bool = False
    horizon: int = 24
    input_size: int = 168
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
    val_size: int = 72
    test_size: int = 72
    random_seed: int = 42
    max_steps_smoke: int = 30
    batch_size: int = 32
    num_workers: int = 0
    output_dir: Path | None = None
    skip_tft: bool = False
    skip_deepar: bool = False


@dataclass(frozen=True)
class AcceleratorInfo:
    kind: str
    device_name: str
    trainer_kwargs: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must parse to mapping: {path}")
    return raw


def load_training_config(
    zone: str,
    *,
    max_steps_smoke: int | None = None,
) -> BaselineTrainingConfig:
    """Load baseline training config from conf yaml files."""
    root = get_project_paths().root
    tr = _load_yaml(root / "conf" / "training.yaml").get("training", {})
    if not isinstance(tr, dict):
        raise ValueError("conf/training.yaml missing training mapping")

    steps = int(max_steps_smoke if max_steps_smoke is not None else tr.get("max_steps_smoke", 30))
    quantiles = tuple(float(q) for q in tr.get("quantiles", [0.1, 0.5, 0.9]))
    if len(quantiles) != 3:
        raise ValueError("Expected exactly 3 quantiles in training config.")

    return BaselineTrainingConfig(
        zone=zone.upper(),
        horizon=int(tr.get("horizon", 24)),
        input_size=int(tr.get("input_size", 168)),
        quantiles=(quantiles[0], quantiles[1], quantiles[2]),
        val_size=int(tr.get("val_size", 72)),
        test_size=int(tr.get("test_size", 72)),
        random_seed=int(tr.get("random_seed", 42)),
        max_steps_smoke=steps,
        batch_size=int(tr.get("batch_size", 32)),
        num_workers=int(tr.get("num_workers", 0)),
    )


def detect_accelerator() -> AcceleratorInfo:
    """Detect available training accelerator and trainer kwargs."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return AcceleratorInfo(
            kind="gpu",
            device_name=name,
            trainer_kwargs={
                "accelerator": "gpu",
                "devices": 1,
                "precision": 32,
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            },
        )

    return AcceleratorInfo(
        kind="cpu",
        device_name="CPU",
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "precision": 32,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
        },
    )


def _resolve_panel_path(cfg: BaselineTrainingConfig) -> Path:
    root = get_project_paths().root
    default_path = (
        root
        / "data"
        / "processed"
        / "panel"
        / "single_zone"
        / f"{cfg.zone}_panel.parquet"
    )
    return cfg.panel_path or default_path


def _load_or_build_panel(cfg: BaselineTrainingConfig) -> tuple[pd.DataFrame, str, Path]:
    panel_path = _resolve_panel_path(cfg)
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        source_label = str(panel.get("source_label", pd.Series(["unknown"])).iloc[0])
        return panel, source_label, panel_path

    if not cfg.build_panel_if_missing:
        raise FileNotFoundError(
            f"Panel not found at {panel_path}. Use --build-panel-if-missing to create one."
        )

    if not cfg.allow_synthetic_panel:
        raise FileNotFoundError(
            "Panel missing and synthetic fallback disabled. "
            "Use --allow-synthetic-panel with --build-panel-if-missing."
        )

    synth_history = cfg.input_size + cfg.val_size + cfg.test_size + 48
    panel_cfg = PanelBuildConfig(
        zone=cfg.zone,
        allow_synthetic_lmp=True,
        allow_synthetic_weather=True,
        drop_warmup_rows=False,
        min_history_hours=synth_history,
    )
    panel = build_single_zone_panel(panel_cfg)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    return panel, "synthetic", panel_path


def _model_output_dirs(cfg: BaselineTrainingConfig) -> tuple[Path, Path, Path]:
    root = get_project_paths().root
    artifact_root = cfg.output_dir or (root / "artifacts" / "baselines")
    forecast_dir = root / "data" / "cache" / "forecasts"
    report_dir = root / "data" / "cache" / "reports"
    artifact_root.mkdir(parents=True, exist_ok=True)
    forecast_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return artifact_root, forecast_dir, report_dir


def _fit_predict_model(
    model_name: str,
    model: Any,
    train_df: pd.DataFrame,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: BaselineTrainingConfig,
) -> pd.DataFrame:
    nf = NeuralForecast(models=[model], freq="h")
    nf.fit(df=train_df, val_size=0)

    history = history_df[["unique_id", "ds", "y"]].copy().reset_index(drop=True)
    remaining = test_df[["unique_id", "ds", "y"]].copy().reset_index(drop=True)
    pred_chunks: list[pd.DataFrame] = []

    while len(remaining) > 0:
        chunk = min(cfg.horizon, len(remaining))
        expected = remaining.iloc[:chunk][["unique_id", "ds"]].copy()

        raw_pred = nf.predict(df=history, quantiles=[0.1, 0.5, 0.9])
        if "unique_id" not in raw_pred.columns:
            raw_pred = raw_pred.reset_index(drop=False)
        aligned = raw_pred.merge(expected, on=["unique_id", "ds"], how="right")
        pred_chunks.append(aligned)

        history = pd.concat([history, remaining.iloc[:chunk]], ignore_index=True)
        remaining = remaining.iloc[chunk:].reset_index(drop=True)

    pred = pd.concat(pred_chunks, ignore_index=True)
    norm = normalize_neuralforecast_output(pred, model=model_name, actuals=test_df)
    validate_quantile_forecast(norm)
    return norm


def train_tft_baseline(
    train_df: pd.DataFrame,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: BaselineTrainingConfig,
    accel: AcceleratorInfo,
) -> pd.DataFrame:
    """Train TFT baseline and return normalized quantile forecast."""
    root = get_project_paths().root
    tft_cfg = _load_yaml(root / "conf" / "model_tft.yaml").get("model", {})
    if not isinstance(tft_cfg, dict):
        raise ValueError("conf/model_tft.yaml missing model mapping")

    model = TFT(
        h=cfg.horizon,
        input_size=cfg.input_size,
        hidden_size=int(tft_cfg.get("hidden_size", 64)),
        n_head=int(tft_cfg.get("n_head", 4)),
        dropout=float(tft_cfg.get("dropout", 0.1)),
        learning_rate=float(tft_cfg.get("learning_rate", 0.001)),
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
        valid_loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
        max_steps=cfg.max_steps_smoke,
        batch_size=cfg.batch_size,
        random_seed=cfg.random_seed,
        dataloader_kwargs={"num_workers": cfg.num_workers},
        **accel.trainer_kwargs,
    )
    return _fit_predict_model("TFT", model, train_df, history_df, test_df, cfg)


def train_deepar_baseline(
    train_df: pd.DataFrame,
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: BaselineTrainingConfig,
    accel: AcceleratorInfo,
) -> pd.DataFrame:
    """Train DeepAR benchmark and return normalized quantile forecast."""
    root = get_project_paths().root
    dr_cfg = _load_yaml(root / "conf" / "model_deepar.yaml").get("model", {})
    if not isinstance(dr_cfg, dict):
        raise ValueError("conf/model_deepar.yaml missing model mapping")

    dist_loss = DistributionLoss(
        distribution="StudentT",
        quantiles=[0.1, 0.5, 0.9],
        level=[80],
    )

    model = DeepAR(
        h=cfg.horizon,
        input_size=cfg.input_size,
        lstm_hidden_size=int(dr_cfg.get("lstm_hidden_size", 64)),
        lstm_n_layers=int(dr_cfg.get("lstm_n_layers", 2)),
        lstm_dropout=float(dr_cfg.get("lstm_dropout", 0.1)),
        learning_rate=float(dr_cfg.get("learning_rate", 0.001)),
        loss=dist_loss,
        valid_loss=dist_loss,
        max_steps=cfg.max_steps_smoke,
        batch_size=cfg.batch_size,
        random_seed=cfg.random_seed,
        dataloader_kwargs={"num_workers": cfg.num_workers},
        **accel.trainer_kwargs,
    )
    return _fit_predict_model("DeepAR", model, train_df, history_df, test_df, cfg)


def train_single_zone_baselines(cfg: BaselineTrainingConfig) -> dict[str, Any]:
    """Train baseline models and persist forecast/metric/report artifacts."""
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    panel, source_label, panel_path = _load_or_build_panel(cfg)
    try:
        validate_panel_schema(panel, PanelBuildConfig(zone=cfg.zone))
    except ValueError as exc:
        should_rebuild_synth = cfg.allow_synthetic_panel and cfg.build_panel_if_missing
        if not should_rebuild_synth:
            raise ValueError(
                "Panel schema validation failed. Rebuild panel or enable synthetic fallback."
            ) from exc

        synth_history = cfg.input_size + cfg.val_size + cfg.test_size + 48
        panel_cfg = PanelBuildConfig(
            zone=cfg.zone,
            allow_synthetic_lmp=True,
            allow_synthetic_weather=True,
            drop_warmup_rows=False,
            min_history_hours=synth_history,
        )
        panel = build_single_zone_panel(panel_cfg)
        panel.to_parquet(panel_path, index=False)
        source_label = "synthetic"

    split_cfg = TimeSplitConfig(val_size=cfg.val_size, test_size=cfg.test_size)
    train_df, val_df, test_df = split_single_series_panel(
        panel[["unique_id", "ds", "y"]],
        split_cfg,
    )
    history_df = pd.concat([train_df, val_df], ignore_index=True)

    accel = detect_accelerator()
    artifact_dir, forecast_dir, report_dir = _model_output_dirs(cfg)

    outputs: dict[str, Any] = {
        "zone": cfg.zone,
        "panel_path": str(panel_path),
        "data_source_label": source_label,
        "accelerator": accel.kind,
        "device_name": accel.device_name,
        "forecasts": {},
        "metrics": {},
        "artifact_dir": str(artifact_dir),
    }

    if not cfg.skip_tft:
        tft_fcst = train_tft_baseline(train_df, history_df, test_df, cfg, accel)
        tft_path = forecast_dir / f"tft_forecast_{cfg.zone}.parquet"
        tft_fcst.to_parquet(tft_path, index=False)
        (artifact_dir / f"tft_artifact_{cfg.zone}.json").write_text(
            json.dumps(
                {
                    "model": "TFT",
                    "zone": cfg.zone,
                    "max_steps": cfg.max_steps_smoke,
                    "generated_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        outputs["forecasts"]["TFT"] = str(tft_path)
        outputs["metrics"]["TFT"] = evaluate_probabilistic_forecast(
            tft_fcst,
            model="TFT",
            zone=cfg.zone,
            data_source_label=source_label,
        )

    if not cfg.skip_deepar:
        dr_fcst = train_deepar_baseline(train_df, history_df, test_df, cfg, accel)
        dr_path = forecast_dir / f"deepar_forecast_{cfg.zone}.parquet"
        dr_fcst.to_parquet(dr_path, index=False)
        (artifact_dir / f"deepar_artifact_{cfg.zone}.json").write_text(
            json.dumps(
                {
                    "model": "DeepAR",
                    "zone": cfg.zone,
                    "max_steps": cfg.max_steps_smoke,
                    "generated_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        outputs["forecasts"]["DeepAR"] = str(dr_path)
        outputs["metrics"]["DeepAR"] = evaluate_probabilistic_forecast(
            dr_fcst,
            model="DeepAR",
            zone=cfg.zone,
            data_source_label=source_label,
        )

    metrics_json = report_dir / f"baseline_metrics_{cfg.zone}.json"
    metrics_csv = report_dir / f"baseline_metrics_{cfg.zone}.csv"
    metrics_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "zone": cfg.zone,
        "data_source_label": source_label,
        "accelerator": {"kind": accel.kind, "device_name": accel.device_name},
        "metrics": outputs["metrics"],
    }
    metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if outputs["metrics"]:
        pd.DataFrame(list(outputs["metrics"].values())).to_csv(metrics_csv, index=False)

    train_report = report_dir / f"baseline_training_report_{cfg.zone}.json"
    train_report.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "zone": cfg.zone,
                "panel_path": str(panel_path),
                "source_label": source_label,
                "split_sizes": {
                    "train": len(train_df),
                    "val": len(val_df),
                    "test": len(test_df),
                },
                "accelerator": {"kind": accel.kind, "device_name": accel.device_name},
                "forecasts": outputs["forecasts"],
                "metrics_json": str(metrics_json),
                "metrics_csv": str(metrics_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    outputs["metrics_json"] = str(metrics_json)
    outputs["metrics_csv"] = str(metrics_csv)
    outputs["training_report_json"] = str(train_report)
    return outputs
