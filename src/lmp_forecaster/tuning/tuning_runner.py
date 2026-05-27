"""Focused tuning execution workflow for single-zone AEP baselines."""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.eval.backtest_runner import BacktestRunConfig, run_rolling_backtest
from lmp_forecaster.eval.calibration import (
    CalibrationDiagnosticConfig,
    discover_latest_backtest_outputs,
    summarize_calibration_diagnostics,
)
from lmp_forecaster.models.baselines import (
    AcceleratorInfo,
    BaselineTrainingConfig,
    detect_accelerator,
    train_deepar_baseline,
    train_tft_baseline,
)
from lmp_forecaster.tracking.mlflow_utils import (
    TrackingConfig,
    configure_mlflow,
    log_artifact_paths,
    log_training_config,
    start_mlflow_run,
)
from lmp_forecaster.tuning.promotion import (
    BaselineMetrics,
    CandidateMetrics,
    PromotionDecision,
    PromotionGate,
    evaluate_candidate_against_baseline,
    summarize_promotion_decisions,
)


@dataclass(frozen=True)
class ResourceProfile:
    name: str = "local_8gb_vram_16gb_ram"
    max_trials_safe: int = 2
    folds_safe: int = 1
    max_steps_cap_safe: int = 3
    batch_size_safe: int = 4
    num_workers: int = 0
    prefer_gpu: bool = True
    allow_cpu_fallback: bool = True
    disable_checkpoints: bool = True
    disable_rich_progress: bool = True
    cleanup_after_trial: bool = True
    max_disk_gb_for_generated_outputs: float = 5.0
    full_search_deferred: bool = True


@dataclass(frozen=True)
class TuningRunConfig:
    zone: str = "AEP"
    panel_path: Path = Path("data/processed/panel/single_zone/AEP_panel.parquet")
    search_design_path: Path | None = None
    baseline_metrics_path: Path | None = None
    models: tuple[str, ...] = ("TFT", "DeepAR")
    max_trials: int = 12
    folds: int = 2
    horizon_hours: int = 24
    min_train_hours: int = 2160
    window_mode: str = "expanding"
    primary_metric: str = "mean_pinball_loss"
    target_coverage: float = 0.80
    coverage_min: float = 0.70
    coverage_max: float = 0.90
    mae_regression_limit: float = 0.15
    allow_deepar_if_interval_collapse: bool = False
    skip_tft: bool = False
    skip_deepar: bool = False
    seed: int = 42
    max_steps_cap: int = 60
    batch_size: int | None = None
    num_workers: int = 0
    cpu_only: bool = False
    cleanup_after_trial: bool = True
    resource_profile: str | None = None
    allow_heavy_run: bool = False
    use_optuna: bool = False
    optuna_storage_path: Path | None = None
    enable_tracking: bool = False
    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    timeout_minutes: int | None = None
    output_root: Path = Path("data/cache/tuning")
    report_root: Path = Path("data/cache/reports")
    artifact_root: Path = Path("artifacts/tuning")
    profile: ResourceProfile = ResourceProfile()

    def validate(self) -> None:
        if self.max_trials <= 0:
            raise ValueError("max_trials must be > 0")
        if self.folds <= 0:
            raise ValueError("folds must be > 0")
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be > 0")
        if self.min_train_hours <= 0:
            raise ValueError("min_train_hours must be > 0")
        if self.window_mode not in {"expanding", "rolling"}:
            raise ValueError("window_mode must be expanding or rolling")
        if not (0.0 < self.coverage_min < self.target_coverage < self.coverage_max < 1.0):
            raise ValueError("coverage_min < target_coverage < coverage_max must hold in (0,1)")
        if self.mae_regression_limit < 0:
            raise ValueError("mae_regression_limit must be >= 0")
        if self.max_steps_cap <= 0:
            raise ValueError("max_steps_cap must be > 0")
        if self.timeout_minutes is not None and self.timeout_minutes <= 0:
            raise ValueError("timeout_minutes must be > 0 when provided")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be > 0 when provided")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")

        supported = {"TFT", "DEEPAR"}
        invalid = sorted({m.upper() for m in self.models if m.upper() not in supported})
        if invalid:
            raise ValueError(f"Unsupported model(s): {invalid}")

        if not self.enabled_models:
            raise ValueError("At least one model must run.")

        if self.resource_profile == "local_safe" and not self.allow_heavy_run:
            if self.max_trials > self.profile.max_trials_safe:
                raise ValueError(
                    "local_safe profile refused heavy run: max_trials exceeds safe limit. "
                    "Use --allow-heavy-run to override."
                )
            if self.folds > self.profile.folds_safe:
                raise ValueError(
                    "local_safe profile refused heavy run: folds exceeds safe limit. "
                    "Use --allow-heavy-run to override."
                )
            if self.max_steps_cap > self.profile.max_steps_cap_safe:
                raise ValueError(
                    "local_safe profile refused heavy run: max_steps_cap exceeds safe limit. "
                    "Use --allow-heavy-run to override."
                )
            if self.batch_size is not None and self.batch_size > self.profile.batch_size_safe:
                raise ValueError(
                    "local_safe profile refused heavy run: batch_size exceeds safe limit. "
                    "Use --allow-heavy-run to override."
                )

    @property
    def enabled_models(self) -> list[str]:
        out: list[str] = []
        for model in self.models:
            key = model.upper()
            if key == "TFT" and self.skip_tft:
                continue
            if key == "DEEPAR" and self.skip_deepar:
                continue
            out.append("DeepAR" if key == "DEEPAR" else "TFT")
        return out

    @property
    def effective_batch_size(self) -> int:
        if self.batch_size is not None:
            return self.batch_size
        if self.resource_profile == "local_safe":
            return self.profile.batch_size_safe
        return 32


@dataclass(frozen=True)
class TrialConfig:
    trial_id: str
    model: str
    config: dict[str, Any]
    source: str


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    model: str
    zone: str
    folds_completed: int
    total_test_rows: int
    config: dict[str, Any]
    MAE_mean: float
    RMSE_mean: float
    mean_pinball_loss_mean: float
    coverage_80_mean: float
    interval_width_mean: float
    median_bias_mean: float | None
    quantile_crossing_rate: float | None
    interval_collapse_warning: bool
    objective_score: float
    promotion_status: str
    rejection_reason: str | None
    runtime_seconds: float
    data_source_label: str
    coverage_gate_passed: bool
    mae_regression_gate_passed: bool
    interval_collapse_gate_passed: bool


@dataclass(frozen=True)
class ModelTuningResult:
    model: str
    trials_requested: int
    trials_completed: int
    best_trial_id: str | None


@dataclass
class TuningSummary:
    config: TuningRunConfig
    trial_results: list[TrialResult]
    model_results: list[ModelTuningResult]
    ranked_candidates: pd.DataFrame
    promotion_decisions: list[PromotionDecision]
    promotion_summary: dict[str, Any]
    search_design_path: str
    baseline_metrics_path: str
    output_paths: dict[str, str] | None = None
    tracking: dict[str, Any] | None = None


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (get_project_paths().root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must parse to mapping: {path}")
    return payload


def _profile_from_config(raw: dict[str, Any]) -> ResourceProfile:
    rp = raw.get("resource_profile", {})
    if not isinstance(rp, dict):
        rp = {}
    return ResourceProfile(
        name=str(rp.get("name", "local_8gb_vram_16gb_ram")),
        max_trials_safe=int(rp.get("max_trials_safe", 2)),
        folds_safe=int(rp.get("folds_safe", 1)),
        max_steps_cap_safe=int(rp.get("max_steps_cap_safe", 3)),
        batch_size_safe=int(rp.get("batch_size_safe", 4)),
        num_workers=int(rp.get("num_workers", 0)),
        prefer_gpu=bool(rp.get("prefer_gpu", True)),
        allow_cpu_fallback=bool(rp.get("allow_cpu_fallback", True)),
        disable_checkpoints=bool(rp.get("disable_checkpoints", True)),
        disable_rich_progress=bool(rp.get("disable_rich_progress", True)),
        cleanup_after_trial=bool(rp.get("cleanup_after_trial", True)),
        max_disk_gb_for_generated_outputs=float(rp.get("max_disk_gb_for_generated_outputs", 5)),
        full_search_deferred=bool(rp.get("full_search_deferred", True)),
    )


def load_tuning_config(config_path: Path | None = None) -> TuningRunConfig:
    root = get_project_paths().root
    path = config_path or (root / "conf" / "tuning.yaml")
    raw = _load_yaml(path)
    section = raw.get("tuning", raw)
    if not isinstance(section, dict):
        raise ValueError("conf/tuning.yaml must define a mapping")

    models_raw = section.get("models", ["TFT", "DeepAR"])
    if not isinstance(models_raw, list) or not all(isinstance(v, str) for v in models_raw):
        raise ValueError("tuning.models must be a list of model names")

    max_trials_value = section.get("max_trials_first_pass")
    if max_trials_value is None:
        max_trials_value = section.get("max_trials", 12)

    folds_value = section.get("folds_for_full_first_pass")
    if folds_value is None:
        folds_value = section.get("folds_for_tuning", 2)

    profile = _profile_from_config(raw)

    cfg = TuningRunConfig(
        zone=str(section.get("zone", "AEP")).upper(),
        panel_path=Path(
            str(
                section.get(
                    "panel_path",
                    "data/processed/panel/single_zone/AEP_panel.parquet",
                )
            )
        ),
        models=tuple(models_raw),
        max_trials=int(max_trials_value),
        folds=int(folds_value),
        horizon_hours=int(section.get("horizon_hours", 24)),
        min_train_hours=int(section.get("min_train_hours", 2160)),
        window_mode=str(section.get("window_mode", "expanding")),
        primary_metric=str(section.get("primary_metric", "mean_pinball_loss")),
        target_coverage=float(section.get("target_coverage", 0.80)),
        coverage_min=float(section.get("coverage_min", 0.70)),
        coverage_max=float(section.get("coverage_max", 0.90)),
        mae_regression_limit=float(section.get("mae_regression_limit", 0.15)),
        allow_deepar_if_interval_collapse=bool(
            section.get("allow_deepar_if_interval_collapse", False)
        ),
        seed=int(section.get("seed", 42)),
        max_steps_cap=int(section.get("max_steps_cap", 60)),
        enable_tracking=bool(section.get("enable_tracking", False)),
        timeout_minutes=(
            int(timeout_raw)
            if (timeout_raw := section.get("timeout_minutes")) is not None
            else None
        ),
        output_root=Path(str(section.get("output_root", "data/cache/tuning"))),
        report_root=Path(str(section.get("report_root", "data/cache/reports"))),
        artifact_root=Path(str(section.get("artifact_root", "artifacts/tuning"))),
        profile=profile,
    )
    cfg.validate()
    return cfg


def _latest_file(pattern: str, root: Path) -> Path | None:
    files = list(root.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _discover_latest_search_design(zone: str) -> Path | None:
    reports_root = _resolve_path(Path("data/cache/reports"))
    if not reports_root.exists():
        return None
    return _latest_file(f"{zone.lower()}_focused_search_design_*.json", reports_root)


def _discover_latest_baseline_metrics(zone: str) -> Path | None:
    discovered = discover_latest_backtest_outputs(zone)
    return discovered.get("aggregate_metrics")


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object mapping: {path}")
    return payload


def load_search_design(zone: str, path: Path | None = None) -> tuple[dict[str, Any], Path]:
    resolved = _resolve_path(path) if path is not None else _discover_latest_search_design(zone)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(
            "Focused search design is missing. Run: "
            "uv run python -m lmp_forecaster.cli design-focused-search --zone AEP --write"
        )
    return _load_json_mapping(resolved), resolved


def _load_baseline_metrics_map(path: Path, zone: str) -> dict[str, BaselineMetrics]:
    frame = pd.read_csv(path)
    required = {
        "model",
        "MAE_mean",
        "RMSE_mean",
        "mean_pinball_loss_mean",
        "coverage_80_mean",
        "interval_width_mean",
        "data_source_label",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Baseline metrics file missing columns: {missing}")

    out: dict[str, BaselineMetrics] = {}
    for _, row in frame.iterrows():
        model = str(row["model"])
        out[model] = BaselineMetrics(
            model=model,
            MAE_mean=float(row["MAE_mean"]),
            RMSE_mean=float(row["RMSE_mean"]),
            mean_pinball_loss_mean=float(row["mean_pinball_loss_mean"]),
            coverage_80_mean=float(row["coverage_80_mean"]),
            interval_width_mean=float(row["interval_width_mean"]),
            data_source_label=str(row.get("data_source_label", "real")),
        )

    needed = set(out).intersection({"TFT", "DeepAR"})
    if not needed:
        raise ValueError(f"Baseline metrics file has no rows for zone {zone} models")
    return out


def load_baseline_metrics(
    zone: str,
    path: Path | None = None,
) -> tuple[dict[str, BaselineMetrics], Path]:
    resolved = _resolve_path(path) if path is not None else _discover_latest_baseline_metrics(zone)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(
            "Rolling backtest aggregate metrics are missing. Run: "
            "uv run python -m lmp_forecaster.cli run-rolling-backtest --zone AEP --write"
        )
    return _load_baseline_metrics_map(resolved, zone), resolved


def _split_trial_budget(total: int, models: list[str]) -> dict[str, int]:
    if not models:
        return {}
    counts = {m: total // len(models) for m in models}
    for idx in range(total % len(models)):
        counts[models[idx]] += 1
    return counts


def _coerce_trial_value(value: Any, spec: dict[str, Any], max_steps_cap: int) -> Any:
    kind = str(spec.get("type", "")).lower()
    name = str(spec.get("name", ""))

    if kind == "int":
        parsed = int(round(float(value)))
        if name == "max_steps":
            return int(min(parsed, max_steps_cap))
        return parsed
    if kind in {"float", "float_log"}:
        return float(value)
    return value


def _sample_trials_deterministic(
    model: str,
    specs: list[dict[str, Any]],
    trial_count: int,
    seed: int,
    max_steps_cap: int,
) -> list[TrialConfig]:
    rng = np.random.default_rng(seed + sum(ord(c) for c in model))
    trials: list[TrialConfig] = []

    for idx in range(trial_count):
        cfg: dict[str, Any] = {}
        for spec in specs:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            kind = str(spec.get("type", "")).lower()
            if kind == "categorical":
                values = list(spec.get("values", []))
                if not values:
                    continue
                value = values[idx % len(values)]
            elif kind == "int":
                lower = int(spec.get("min", 1))
                upper = int(spec.get("max", lower))
                value = int(rng.integers(lower, upper + 1))
            elif kind == "float_log":
                lower_f = float(spec.get("min", 1e-4))
                upper_f = float(spec.get("max", 1.0))
                value = float(math.exp(rng.uniform(math.log(lower_f), math.log(upper_f))))
            elif kind == "float":
                lower_f = float(spec.get("min", 0.0))
                upper_f = float(spec.get("max", 1.0))
                value = float(rng.uniform(lower_f, upper_f))
            else:
                continue

            cfg[name] = _coerce_trial_value(value, spec, max_steps_cap)

        trials.append(
            TrialConfig(
                trial_id=f"{model.lower()}_{idx + 1:03d}",
                model=model,
                config=cfg,
                source="deterministic-grid",
            )
        )

    return trials


def _sample_trials_optuna(
    model: str,
    specs: list[dict[str, Any]],
    trial_count: int,
    seed: int,
    max_steps_cap: int,
    storage_path: Path | None,
) -> list[TrialConfig]:
    import optuna

    sampler = optuna.samplers.TPESampler(seed=seed + sum(ord(c) for c in model))
    if storage_path is None:
        study = optuna.create_study(direction="minimize", sampler=sampler)
    else:
        storage_uri = f"sqlite:///{_resolve_path(storage_path).as_posix()}"
        study = optuna.create_study(direction="minimize", sampler=sampler, storage=storage_uri)

    trials: list[TrialConfig] = []
    for idx in range(trial_count):
        ask_trial = study.ask()
        cfg: dict[str, Any] = {}
        for spec in specs:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            kind = str(spec.get("type", "")).lower()
            if kind == "categorical":
                values = list(spec.get("values", []))
                if not values:
                    continue
                value = ask_trial.suggest_categorical(name, values)
            elif kind == "int":
                value = ask_trial.suggest_int(
                    name,
                    int(spec.get("min", 1)),
                    int(spec.get("max", 1)),
                )
            elif kind == "float_log":
                value = ask_trial.suggest_float(
                    name,
                    float(spec.get("min", 1e-4)),
                    float(spec.get("max", 1.0)),
                    log=True,
                )
            elif kind == "float":
                value = ask_trial.suggest_float(
                    name,
                    float(spec.get("min", 0.0)),
                    float(spec.get("max", 1.0)),
                )
            else:
                continue
            cfg[name] = _coerce_trial_value(value, spec, max_steps_cap)

        trials.append(
            TrialConfig(
                trial_id=f"{model.lower()}_{idx + 1:03d}",
                model=model,
                config=cfg,
                source="optuna-tpe",
            )
        )
        study.tell(ask_trial, float(idx))

    return trials


def build_trial_grid_or_optuna_space(
    search_design: dict[str, Any],
    *,
    model: str,
    trial_count: int,
    seed: int,
    max_steps_cap: int,
    use_optuna: bool = False,
    optuna_storage_path: Path | None = None,
) -> list[TrialConfig]:
    spaces = search_design.get("spaces", {})
    if not isinstance(spaces, dict):
        raise ValueError("Search design missing spaces mapping")
    model_space = spaces.get(model)
    if not isinstance(model_space, dict):
        raise ValueError(f"Search design missing model space for {model}")

    specs = model_space.get("parameters", [])
    if not isinstance(specs, list):
        raise ValueError(f"Search design model parameters must be list: {model}")

    if use_optuna:
        try:
            import optuna  # noqa: F401

            return _sample_trials_optuna(
                model,
                specs,
                trial_count,
                seed,
                max_steps_cap,
                storage_path=optuna_storage_path,
            )
        except Exception:
            return _sample_trials_deterministic(model, specs, trial_count, seed, max_steps_cap)

    return _sample_trials_deterministic(model, specs, trial_count, seed, max_steps_cap)


def _sanitize_trial_config(model: str, config: dict[str, Any]) -> dict[str, Any]:
    """Apply lightweight model-specific constraints to avoid invalid trial configs."""
    sanitized = dict(config)
    if model == "TFT" and "hidden_size" in sanitized:
        n_head = int(sanitized.get("n_head", 4))
        hidden_size = int(sanitized["hidden_size"])
        if n_head <= 0:
            n_head = 1
        remainder = hidden_size % n_head
        if remainder != 0:
            sanitized["hidden_size"] = hidden_size + (n_head - remainder)
    return sanitized


def _resolve_trial_accelerator(
    run_cfg: TuningRunConfig,
    fallback: AcceleratorInfo,
) -> AcceleratorInfo:
    if run_cfg.cpu_only:
        return detect_accelerator("cpu")
    if run_cfg.resource_profile == "local_safe":
        if run_cfg.profile.prefer_gpu:
            try:
                return detect_accelerator("gpu")
            except Exception:
                if run_cfg.profile.allow_cpu_fallback:
                    return detect_accelerator("cpu")
                raise
        return detect_accelerator("cpu")
    return fallback


def _trial_model_runner(
    model_overrides: dict[str, Any],
    run_cfg: TuningRunConfig,
) -> Callable[
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
]:
    def _runner(
        model: str,
        train_df: pd.DataFrame,
        history_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_cfg: BaselineTrainingConfig,
        accelerator: AcceleratorInfo,
        data_source_label: str,
    ) -> pd.DataFrame:
        cfg_kwargs = {**train_cfg.__dict__}
        sanitized_overrides = _sanitize_trial_config(model, model_overrides)
        if "input_size_hours" in sanitized_overrides:
            cfg_kwargs["input_size"] = int(sanitized_overrides["input_size_hours"])
        override_batch_size = (
            int(sanitized_overrides["batch_size"])
            if "batch_size" in sanitized_overrides
            else int(run_cfg.effective_batch_size)
        )
        if run_cfg.resource_profile == "local_safe" and not run_cfg.allow_heavy_run:
            override_batch_size = min(override_batch_size, run_cfg.profile.batch_size_safe)
        cfg_kwargs["batch_size"] = override_batch_size
        if "max_steps" in sanitized_overrides:
            cfg_kwargs["max_steps_smoke"] = int(sanitized_overrides["max_steps"])
        cfg_kwargs["num_workers"] = int(run_cfg.num_workers)

        tuned_cfg = BaselineTrainingConfig(**cfg_kwargs)
        trial_accel = _resolve_trial_accelerator(run_cfg, accelerator)

        if model == "TFT":
            return train_tft_baseline(
                train_df,
                history_df,
                test_df,
                tuned_cfg,
                trial_accel,
                data_source_label=data_source_label,
                model_overrides=sanitized_overrides,
            )
        if model == "DeepAR":
            return train_deepar_baseline(
                train_df,
                history_df,
                test_df,
                tuned_cfg,
                trial_accel,
                data_source_label=data_source_label,
                model_overrides=sanitized_overrides,
            )
        raise ValueError(f"Unsupported model for tuning runner: {model}")

    return _runner


def _build_backtest_config_for_trial(run_cfg: TuningRunConfig, model: str) -> BacktestRunConfig:
    return BacktestRunConfig(
        zone=run_cfg.zone,
        panel_path=run_cfg.panel_path,
        folds=run_cfg.folds,
        horizon_hours=run_cfg.horizon_hours,
        min_train_hours=run_cfg.min_train_hours,
        window_mode=run_cfg.window_mode,
        models=(model,),
        max_steps=run_cfg.max_steps_cap,
        seed=run_cfg.seed,
        skip_tft=model != "TFT",
        skip_deepar=model != "DeepAR",
        require_real_data=True,
        enable_tracking=False,
    )


def _safe_remove(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            return


def cleanup_after_trial(run_cfg: TuningRunConfig, trial_id: str) -> None:
    if run_cfg.cleanup_after_trial:
        root = get_project_paths().root
        candidates = [
            _resolve_path(run_cfg.artifact_root) / trial_id,
            root / "lightning_logs",
            root / "checkpoints",
            root / "artifacts" / "tuning" / trial_id,
        ]
        for candidate in candidates:
            _safe_remove(candidate)

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    if "cuda" in message and "out of memory" in message:
        return True
    try:
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except Exception:
        return False


def run_single_tuning_trial(
    trial: TrialConfig,
    run_cfg: TuningRunConfig,
    baseline_map: dict[str, BaselineMetrics],
) -> tuple[TrialResult, PromotionDecision]:
    t0 = time.perf_counter()

    if trial.model not in baseline_map:
        raise ValueError(f"Missing baseline metrics for model {trial.model}")

    backtest_cfg = _build_backtest_config_for_trial(run_cfg, trial.model)
    trial_runner = _trial_model_runner(trial.config, run_cfg)

    result = run_rolling_backtest(backtest_cfg, model_runner=trial_runner)
    model_row = (
        result.aggregate_metrics[result.aggregate_metrics["model"] == trial.model]
        .reset_index(drop=True)
        .iloc[0]
    )

    forecast_model = result.forecasts[result.forecasts["model"] == trial.model].copy()
    cal_cfg = CalibrationDiagnosticConfig(target_coverage=run_cfg.target_coverage)
    diagnostics = summarize_calibration_diagnostics(
        forecast_model,
        cfg=cal_cfg,
        zone=run_cfg.zone,
    )

    diag_rows = [
        row for row in diagnostics.get("model_summary", []) if str(row.get("model")) == trial.model
    ]
    diag = diag_rows[0] if diag_rows else {}

    candidate = CandidateMetrics(
        model=trial.model,
        trial_id=trial.trial_id,
        MAE_mean=float(model_row["MAE_mean"]),
        RMSE_mean=float(model_row["RMSE_mean"]),
        mean_pinball_loss_mean=float(model_row["mean_pinball_loss_mean"]),
        coverage_80_mean=float(model_row["coverage_80_mean"]),
        interval_width_mean=float(model_row["interval_width_mean"]),
        median_bias_mean=(
            float(median_bias_raw)
            if (median_bias_raw := diag.get("median_bias_mean")) is not None
            else None
        ),
        quantile_crossing_rate=(
            float(crossing_raw) if (crossing_raw := diag.get("crossing_rate")) is not None else None
        ),
        interval_collapse_warning=bool(diag.get("interval_collapse_warning", False)),
    )

    gate = PromotionGate(
        coverage_min=run_cfg.coverage_min,
        coverage_max=run_cfg.coverage_max,
        mae_regression_limit=run_cfg.mae_regression_limit,
        require_no_interval_collapse=not run_cfg.allow_deepar_if_interval_collapse,
    )
    decision = evaluate_candidate_against_baseline(candidate, baseline_map[trial.model], gate)

    promotion_status = decision.promotion_status
    rejection_reason = decision.rejection_reason

    if run_cfg.folds <= 1 and decision.promoted:
        promotion_status = "smoke_candidate_requires_full_validation"
        rejection_reason = "single-fold smoke tuning result requires multi-fold validation"

    elapsed = time.perf_counter() - t0
    trial_result = TrialResult(
        trial_id=trial.trial_id,
        model=trial.model,
        zone=run_cfg.zone,
        folds_completed=int(model_row["folds_completed"]),
        total_test_rows=int(model_row["total_test_rows"]),
        config=trial.config,
        MAE_mean=float(model_row["MAE_mean"]),
        RMSE_mean=float(model_row["RMSE_mean"]),
        mean_pinball_loss_mean=float(model_row["mean_pinball_loss_mean"]),
        coverage_80_mean=float(model_row["coverage_80_mean"]),
        interval_width_mean=float(model_row["interval_width_mean"]),
        median_bias_mean=candidate.median_bias_mean,
        quantile_crossing_rate=candidate.quantile_crossing_rate,
        interval_collapse_warning=candidate.interval_collapse_warning,
        objective_score=decision.objective_score,
        promotion_status=promotion_status,
        rejection_reason=rejection_reason,
        runtime_seconds=float(elapsed),
        data_source_label=str(model_row["data_source_label"]),
        coverage_gate_passed=decision.coverage_gate_passed,
        mae_regression_gate_passed=decision.mae_regression_gate_passed,
        interval_collapse_gate_passed=decision.interval_collapse_gate_passed,
    )
    return trial_result, decision


def rank_tuning_trials(trials: list[TrialResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    status_rank = {
        "promoted": 0,
        "smoke_candidate_requires_full_validation": 1,
        "rejected": 2,
        "no_promotion": 3,
        "failed_resource_limited": 4,
        "failed": 5,
    }

    for row in trials:
        rows.append(
            {
                **asdict(row),
                "config": json.dumps(row.config, sort_keys=True),
                "status_rank": status_rank.get(row.promotion_status, 9),
            }
        )

    if not rows:
        return pd.DataFrame()

    ranked = (
        pd.DataFrame(rows)
        .sort_values(["status_rank", "objective_score", "mean_pinball_loss_mean", "MAE_mean"])
        .reset_index(drop=True)
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked.drop(columns=["status_rank"])


def decide_promotion(
    trials: list[TrialResult],
    decisions: list[PromotionDecision],
) -> dict[str, Any]:
    summary = summarize_promotion_decisions(decisions)
    if not trials:
        summary["best_candidate"] = None
        summary["overall_status"] = "failed_resource_limited"
        summary["summary_reason"] = "All trials failed before candidate evaluation."
        return summary

    ranked = rank_tuning_trials(trials)
    if ranked.empty:
        summary["best_candidate"] = None
        summary["overall_status"] = "no_promotion"
        summary["summary_reason"] = "No ranked candidates available."
        return summary

    best = ranked.iloc[0].to_dict()
    summary["best_candidate"] = {
        "model": best.get("model"),
        "trial_id": best.get("trial_id"),
        "promotion_status": best.get("promotion_status"),
        "rejection_reason": best.get("rejection_reason"),
        "objective_score": float(best.get("objective_score", 0.0)),
    }

    failed_count = int((ranked["promotion_status"] == "failed_resource_limited").sum())
    if failed_count == len(ranked):
        summary["overall_status"] = "failed_resource_limited"
        summary["summary_reason"] = "All trials failed due to hardware/resource limits."
        return summary

    if best.get("promotion_status") == "smoke_candidate_requires_full_validation":
        summary["overall_status"] = "smoke_candidate_requires_full_validation"
        summary["summary_reason"] = (
            "Best candidate improved in smoke scope but full validation is required "
            "before promotion."
        )
    elif best.get("promotion_status") != "promoted":
        summary["overall_status"] = "no_promotion"

    return summary


def _model_tuning_rows(trials: list[TrialResult], model: str, requested: int) -> ModelTuningResult:
    subset = [row for row in trials if row.model == model]
    if subset:
        best = min(subset, key=lambda row: row.objective_score)
        best_id = best.trial_id
    else:
        best_id = None

    completed = len(
        [
            row
            for row in subset
            if row.promotion_status not in {"failed", "failed_resource_limited"}
        ]
    )
    return ModelTuningResult(
        model=model,
        trials_requested=requested,
        trials_completed=completed,
        best_trial_id=best_id,
    )


def run_focused_tuning(
    run_cfg: TuningRunConfig,
    *,
    search_design: dict[str, Any] | None = None,
    baseline_map: dict[str, BaselineMetrics] | None = None,
    search_design_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
) -> TuningSummary:
    run_cfg.validate()

    loaded_design: dict[str, Any]
    loaded_baselines: dict[str, BaselineMetrics]

    if search_design is not None:
        loaded_design = search_design
        design_path_text = (
            str(_resolve_path(search_design_path))
            if search_design_path is not None
            else "<in_memory_search_design>"
        )
    else:
        loaded_design, discovered_design_path = load_search_design(
            run_cfg.zone,
            search_design_path or run_cfg.search_design_path,
        )
        design_path_text = str(discovered_design_path)

    if baseline_map is not None:
        loaded_baselines = baseline_map
        baseline_path_text = (
            str(_resolve_path(baseline_metrics_path))
            if baseline_metrics_path is not None
            else "<in_memory_baseline_metrics>"
        )
    else:
        loaded_baselines, discovered_baseline_path = load_baseline_metrics(
            run_cfg.zone,
            baseline_metrics_path or run_cfg.baseline_metrics_path,
        )
        baseline_path_text = str(discovered_baseline_path)

    budgets = _split_trial_budget(run_cfg.max_trials, run_cfg.enabled_models)
    all_trials: list[TrialConfig] = []
    for model in run_cfg.enabled_models:
        trial_count = budgets.get(model, 0)
        if trial_count <= 0:
            continue
        model_trials = build_trial_grid_or_optuna_space(
            loaded_design,
            model=model,
            trial_count=trial_count,
            seed=run_cfg.seed,
            max_steps_cap=run_cfg.max_steps_cap,
            use_optuna=run_cfg.use_optuna,
            optuna_storage_path=run_cfg.optuna_storage_path,
        )
        all_trials.extend(model_trials)

    ordered_trials = sorted(all_trials, key=lambda row: row.trial_id)

    trial_results: list[TrialResult] = []
    decisions: list[PromotionDecision] = []
    run_started = time.perf_counter()

    for trial in ordered_trials:
        trial_t0 = time.perf_counter()
        if run_cfg.timeout_minutes is not None:
            elapsed_minutes = (time.perf_counter() - run_started) / 60.0
            if elapsed_minutes >= run_cfg.timeout_minutes:
                break
        try:
            trial_result, decision = run_single_tuning_trial(trial, run_cfg, loaded_baselines)
            trial_results.append(trial_result)
            decisions.append(decision)
        except Exception as exc:
            elapsed = time.perf_counter() - trial_t0
            resource_failure = _is_cuda_oom(exc) or "out of memory" in str(exc).lower()
            failure_status = "failed_resource_limited" if resource_failure else "failed"
            reason = str(exc)
            if resource_failure and "cuda" in reason.lower():
                reason = f"cuda_oom: {reason}"
            trial_results.append(
                TrialResult(
                    trial_id=trial.trial_id,
                    model=trial.model,
                    zone=run_cfg.zone,
                    folds_completed=0,
                    total_test_rows=0,
                    config=trial.config,
                    MAE_mean=float("nan"),
                    RMSE_mean=float("nan"),
                    mean_pinball_loss_mean=float("nan"),
                    coverage_80_mean=float("nan"),
                    interval_width_mean=float("nan"),
                    median_bias_mean=None,
                    quantile_crossing_rate=None,
                    interval_collapse_warning=True,
                    objective_score=float("inf"),
                    promotion_status=failure_status,
                    rejection_reason=reason,
                    runtime_seconds=float(elapsed),
                    data_source_label="real",
                    coverage_gate_passed=False,
                    mae_regression_gate_passed=False,
                    interval_collapse_gate_passed=False,
                )
            )
        finally:
            cleanup_after_trial(run_cfg, trial.trial_id)

    ranked = rank_tuning_trials(trial_results)
    promotion_summary = decide_promotion(trial_results, decisions)

    model_results = [
        _model_tuning_rows(trial_results, model, budgets.get(model, 0))
        for model in run_cfg.enabled_models
    ]

    return TuningSummary(
        config=run_cfg,
        trial_results=trial_results,
        model_results=model_results,
        ranked_candidates=ranked,
        promotion_decisions=decisions,
        promotion_summary=promotion_summary,
        search_design_path=design_path_text,
        baseline_metrics_path=baseline_path_text,
    )


def planned_tuning_output_paths(
    cfg: TuningRunConfig,
    stamp: str = "<timestamp>",
) -> dict[str, Path]:
    zone = cfg.zone.lower()
    tuning_root = _resolve_path(cfg.output_root)
    report_root = _resolve_path(cfg.report_root)
    artifact_root = _resolve_path(cfg.artifact_root)
    return {
        "trials": tuning_root / f"{zone}_focused_tuning_trials_{stamp}.csv",
        "ranked": tuning_root / f"{zone}_focused_tuning_ranked_{stamp}.csv",
        "summary_json": report_root / f"{zone}_focused_tuning_summary_{stamp}.json",
        "summary_markdown": report_root / f"{zone}_focused_tuning_summary_{stamp}.md",
        "manifest": artifact_root / f"{zone}_focused_tuning_manifest_{stamp}.json",
    }


def write_tuning_results(summary: TuningSummary) -> dict[str, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cfg = summary.config

    output_root = _resolve_path(cfg.output_root)
    report_root = _resolve_path(cfg.report_root)
    artifact_root = _resolve_path(cfg.artifact_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    paths = planned_tuning_output_paths(cfg, stamp)

    trials_df = pd.DataFrame([asdict(row) for row in summary.trial_results])
    if not trials_df.empty:
        trials_df["config"] = trials_df["config"].map(lambda d: json.dumps(d, sort_keys=True))
    trials_df.to_csv(paths["trials"], index=False)

    ranked = summary.ranked_candidates.copy()
    if not ranked.empty:
        ranked.to_csv(paths["ranked"], index=False)
    else:
        pd.DataFrame().to_csv(paths["ranked"], index=False)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "zone": cfg.zone,
        "models_attempted": cfg.enabled_models,
        "max_trials": cfg.max_trials,
        "folds": cfg.folds,
        "horizon_hours": cfg.horizon_hours,
        "objective": cfg.primary_metric,
        "resource_profile": {
            "name": cfg.profile.name,
            "active": cfg.resource_profile == "local_safe",
            "max_trials_safe": cfg.profile.max_trials_safe,
            "folds_safe": cfg.profile.folds_safe,
            "max_steps_cap_safe": cfg.profile.max_steps_cap_safe,
            "batch_size_safe": cfg.profile.batch_size_safe,
            "num_workers": cfg.num_workers,
            "full_search_deferred": cfg.profile.full_search_deferred,
        },
        "hardware_limited_caveat": (
            "smoke tuning executed on local constrained hardware; full validation deferred"
            if cfg.resource_profile == "local_safe"
            else "none"
        ),
        "promotion_gate": {
            "coverage_min": cfg.coverage_min,
            "coverage_max": cfg.coverage_max,
            "mae_regression_limit": cfg.mae_regression_limit,
            "allow_deepar_if_interval_collapse": cfg.allow_deepar_if_interval_collapse,
        },
        "search_design_path": summary.search_design_path,
        "baseline_metrics_path": summary.baseline_metrics_path,
        "model_results": [asdict(item) for item in summary.model_results],
        "promotion_summary": summary.promotion_summary,
        "best_candidates": summary.ranked_candidates.head(5).to_dict(orient="records"),
        "output_paths": {k: str(v) for k, v in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ranked_md = (
        summary.ranked_candidates.head(10).to_markdown(index=False)
        if not summary.ranked_candidates.empty
        else "(no ranked candidates)"
    )

    md_lines = [
        f"# Focused tuning summary — {cfg.zone}",
        "",
        f"Generated at: {payload['generated_at']}",
        f"Models attempted: {', '.join(cfg.enabled_models)}",
        f"Trials budget: {cfg.max_trials}",
        f"Folds per trial: {cfg.folds}",
        f"Objective: {cfg.primary_metric}",
        f"Resource profile active: {cfg.resource_profile == 'local_safe'}",
        (
            "Promotion gate: "
            f"coverage in [{cfg.coverage_min:.2f}, {cfg.coverage_max:.2f}], "
            f"MAE regression <= {cfg.mae_regression_limit:.2f}, "
            f"allow_deepar_if_interval_collapse={cfg.allow_deepar_if_interval_collapse}"
        ),
        "",
        "## Promotion decision",
        json.dumps(summary.promotion_summary, indent=2),
        "",
        "## Ranked candidates",
        ranked_md,
    ]
    paths["summary_markdown"].write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "zone": cfg.zone,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary.output_paths = {k: str(v) for k, v in paths.items()}
    return paths


def log_tuning_tracking(
    summary: TuningSummary,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    cfg = summary.config
    tracking_cfg = TrackingConfig(
        enabled=cfg.enable_tracking,
        experiment_name=cfg.experiment_name or "lmp_focused_tuning",
        tracking_uri=cfg.tracking_uri or "file:./mlruns",
        run_name_prefix="focused_tuning",
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

    run_name = cfg.run_name or f"focused_tuning_{cfg.zone.lower()}"
    with start_mlflow_run(ctx, run_name=run_name) as run:
        log_training_config(
            run,
            {
                "zone": cfg.zone,
                "models": ",".join(cfg.enabled_models),
                "max_trials": cfg.max_trials,
                "folds": cfg.folds,
                "horizon_hours": cfg.horizon_hours,
                "primary_metric": cfg.primary_metric,
                "coverage_min": cfg.coverage_min,
                "coverage_max": cfg.coverage_max,
                "mae_regression_limit": cfg.mae_regression_limit,
            },
        )

        import mlflow

        promoted = summary.promotion_summary.get("best_promoted_trial")
        if isinstance(promoted, dict):
            score = promoted.get("objective_score")
            if score is not None:
                mlflow.log_metric("best_promoted_objective_score", float(score))

        if not summary.ranked_candidates.empty:
            best_row = summary.ranked_candidates.iloc[0]
            for key in ["MAE_mean", "RMSE_mean", "mean_pinball_loss_mean", "coverage_80_mean"]:
                try:
                    mlflow.log_metric(f"best_candidate_{key}", float(best_row[key]))
                except Exception:
                    pass

        log_artifact_paths(
            run,
            {name: str(path) for name, path in output_paths.items()},
            artifact_file=f"focused_tuning_artifacts_{cfg.zone}.txt",
        )

        info = getattr(run, "info", None)
        status["run_id"] = getattr(info, "run_id", None)
    return status
