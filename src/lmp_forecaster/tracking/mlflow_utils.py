"""Optional MLflow tracking utilities with safe logging guards."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRACKING_UNAVAILABLE_REASONS = {
    "not_installed": "mlflow_not_installed",
    "runtime_error": "mlflow_runtime_error",
    "disabled": "tracking_disabled",
}

_SENSITIVE_KEY_MARKERS = (
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "api",
    "credential",
)


def _is_sensitive_key(name: str) -> bool:
    key = name.lower()
    return any(marker in key for marker in _SENSITIVE_KEY_MARKERS)


def _to_safe_param_value(value: Any) -> str:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    return str(value)


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    experiment_name: str = "lmp_probabilistic_forecaster"
    tracking_uri: str = "file:./mlruns"
    run_name_prefix: str = "baseline"
    log_artifacts: bool = True
    log_model_artifacts: bool = False


@dataclass(frozen=True)
class TrackingContext:
    enabled: bool
    reason: str | None
    config: TrackingConfig


def get_tracking_uri(config: TrackingConfig) -> str:
    return config.tracking_uri.strip() or "file:./mlruns"


def configure_mlflow(config: TrackingConfig) -> TrackingContext:
    if not config.enabled:
        return TrackingContext(
            enabled=False,
            reason=_TRACKING_UNAVAILABLE_REASONS["disabled"],
            config=config,
        )

    try:
        import mlflow
    except ImportError:
        return TrackingContext(
            enabled=False,
            reason=_TRACKING_UNAVAILABLE_REASONS["not_installed"],
            config=config,
        )

    uri = get_tracking_uri(config)
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config.experiment_name)
    except Exception:
        return TrackingContext(
            enabled=False,
            reason=_TRACKING_UNAVAILABLE_REASONS["runtime_error"],
            config=config,
        )

    return TrackingContext(enabled=True, reason=None, config=config)


@contextmanager
def start_mlflow_run(
    context: TrackingContext,
    *,
    run_name: str,
) -> Any:
    if not context.enabled:
        yield None
        return

    import mlflow

    with mlflow.start_run(run_name=run_name) as run:
        yield run


def safe_log_params(run: Any, params: dict[str, Any]) -> dict[str, str]:
    if run is None:
        return {}

    import mlflow

    safe: dict[str, str] = {}
    for key, value in params.items():
        if _is_sensitive_key(key):
            continue
        safe[key] = _to_safe_param_value(value)

    if safe:
        mlflow.log_params(safe)
    return safe


def log_training_config(run: Any, params: dict[str, Any]) -> dict[str, str]:
    return safe_log_params(run, params)


def log_metrics_table(run: Any, metrics: dict[str, dict[str, Any]]) -> None:
    if run is None:
        return

    import mlflow

    flat: dict[str, float] = {}
    for model_name, row in metrics.items():
        prefix = f"{model_name}_"
        for key in ("mae", "rmse", "coverage_80", "mean_pinball_loss"):
            value = row.get(key)
            if value is None:
                continue
            try:
                flat[f"{prefix}{key}"] = float(value)
            except (TypeError, ValueError):
                continue

    if flat:
        mlflow.log_metrics(flat)


def log_artifact_paths(
    run: Any,
    artifacts: dict[str, str],
    *,
    artifact_file: str = "artifact_paths.txt",
) -> Path | None:
    if run is None:
        return None

    import mlflow

    lines = [f"{k}: {Path(v).name}" for k, v in sorted(artifacts.items())]
    if not lines:
        return None

    out = Path(".mlflow_artifacts")
    out.mkdir(parents=True, exist_ok=True)
    txt = out / artifact_file
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(txt))
    return txt
