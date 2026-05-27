"""Focused search design from rolling backtest calibration diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from lmp_forecaster.config.paths import get_project_paths


@dataclass(frozen=True)
class SearchDesignConfig:
    max_trials_first_pass: int = 12
    max_trials_second_pass: int = 30
    timeout_minutes_first_pass: int = 45
    primary_metric: str = "mean_pinball_loss"
    secondary_metric: str = "coverage_80"
    promotion_coverage_min: float = 0.70
    promotion_coverage_max: float = 0.90
    promotion_mae_regression_limit: float = 0.15

    def validate(self) -> None:
        if self.max_trials_first_pass <= 0:
            raise ValueError("max_trials_first_pass must be > 0")
        if self.max_trials_second_pass < self.max_trials_first_pass:
            raise ValueError("max_trials_second_pass must be >= max_trials_first_pass")
        if self.timeout_minutes_first_pass <= 0:
            raise ValueError("timeout_minutes_first_pass must be > 0")
        if not (0.0 < self.promotion_coverage_min < 1.0):
            raise ValueError("promotion_coverage_min must be in (0,1)")
        if not (0.0 < self.promotion_coverage_max < 1.0):
            raise ValueError("promotion_coverage_max must be in (0,1)")
        if self.promotion_coverage_min >= self.promotion_coverage_max:
            raise ValueError("promotion_coverage_min must be < promotion_coverage_max")
        if self.promotion_mae_regression_limit < 0:
            raise ValueError("promotion_mae_regression_limit must be >= 0")


@dataclass(frozen=True)
class ModelSearchSpace:
    model: str
    objective_focus: str
    parameters: list[dict[str, Any]]
    recommended_first_search_size: int
    estimated_cost_level: str
    stop_criteria: list[str]
    promotion_criteria: list[str]


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (get_project_paths().root / path).resolve()


def _latest_file(pattern: str, root: Path) -> Path | None:
    files = list(root.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def load_search_design_config(config_path: Path | None = None) -> SearchDesignConfig:
    root = get_project_paths().root
    path = config_path or (root / "conf" / "search_design.yaml")
    if not path.exists():
        cfg = SearchDesignConfig()
        cfg.validate()
        return cfg

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("search design config must parse to mapping")
    section = payload.get("search_design", payload)
    if not isinstance(section, dict):
        raise ValueError("search_design config section must be mapping")

    cfg = SearchDesignConfig(
        max_trials_first_pass=int(section.get("max_trials_first_pass", 12)),
        max_trials_second_pass=int(section.get("max_trials_second_pass", 30)),
        timeout_minutes_first_pass=int(section.get("timeout_minutes_first_pass", 45)),
        primary_metric=str(section.get("primary_metric", "mean_pinball_loss")),
        secondary_metric=str(section.get("secondary_metric", "coverage_80")),
        promotion_coverage_min=float(section.get("promotion_coverage_min", 0.70)),
        promotion_coverage_max=float(section.get("promotion_coverage_max", 0.90)),
        promotion_mae_regression_limit=float(section.get("promotion_mae_regression_limit", 0.15)),
    )
    cfg.validate()
    return cfg


def discover_latest_calibration_report(
    zone: str = "AEP",
    *,
    report_root: Path = Path("data/cache/reports"),
) -> Path | None:
    root = _resolve_path(report_root)
    if not root.exists():
        return None
    return _latest_file(f"{zone.lower()}_calibration_diagnostics_*.json", root)


def _extract_model_summary_map(diagnostics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = diagnostics.get("model_summary", [])
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model", "")).strip()
        if not model:
            continue
        out[model] = row
    return out


def build_tft_search_space(
    diagnostics: dict[str, Any],
    cfg: SearchDesignConfig,
) -> ModelSearchSpace:
    summary = _extract_model_summary_map(diagnostics).get("TFT", {})
    coverage = float(summary.get("coverage_80", 0.0))

    focus = (
        "Widen predictive intervals while preserving median accuracy; "
        f"observed coverage_80={coverage:.4f} indicates under-coverage."
    )

    params = [
        {
            "name": "dropout",
            "type": "float",
            "min": 0.05,
            "max": 0.35,
            "priority_rank": 1,
            "reason": "Higher regularization can improve uncertainty calibration.",
            "expected_effect": "Potentially wider intervals and reduced overconfidence.",
            "cost_level": "low",
        },
        {
            "name": "learning_rate",
            "type": "float_log",
            "min": 1e-4,
            "max": 3e-3,
            "priority_rank": 2,
            "reason": "Training dynamics affect quantile sharpness/calibration.",
            "expected_effect": "Stabilize quantile learning; avoid overly sharp intervals.",
            "cost_level": "low",
        },
        {
            "name": "hidden_size",
            "type": "int",
            "min": 32,
            "max": 128,
            "priority_rank": 3,
            "reason": "Model capacity tradeoff can improve uncertainty representation.",
            "expected_effect": "Better fit while controlling over/under-dispersion.",
            "cost_level": "medium",
        },
        {
            "name": "input_size_hours",
            "type": "int",
            "min": 96,
            "max": 336,
            "priority_rank": 4,
            "reason": "History window controls regime context for interval estimates.",
            "expected_effect": "Improved multi-horizon calibration and residual profile.",
            "cost_level": "medium",
        },
        {
            "name": "max_steps",
            "type": "int",
            "min": 30,
            "max": 180,
            "priority_rank": 5,
            "reason": "Avoid undertraining-induced quantile bias while staying cost-safe.",
            "expected_effect": "Potential calibration gain at modest runtime increase.",
            "cost_level": "medium",
        },
        {
            "name": "batch_size",
            "type": "categorical",
            "values": [16, 32, 64],
            "priority_rank": 6,
            "reason": "Optimization noise and calibration sensitivity.",
            "expected_effect": "May improve quantile generalization stability.",
            "cost_level": "low",
        },
    ]

    promotion = [
        (
            f"{cfg.secondary_metric} in "
            f"[{cfg.promotion_coverage_min:.2f}, {cfg.promotion_coverage_max:.2f}]"
        ),
        (
            "No more than "
            f"{cfg.promotion_mae_regression_limit:.2f} relative MAE regression "
            "vs Step 8 TFT baseline"
        ),
        f"Primary metric ({cfg.primary_metric}) improved from Step 8 TFT baseline",
    ]

    stop = [
        "Stop first pass after max_trials_first_pass complete",
        "Stop early if 3 consecutive trials worsen primary metric and coverage",
        "Stop if runtime exceeds timeout_minutes_first_pass",
    ]

    return ModelSearchSpace(
        model="TFT",
        objective_focus=focus,
        parameters=params,
        recommended_first_search_size=cfg.max_trials_first_pass,
        estimated_cost_level="medium",
        stop_criteria=stop,
        promotion_criteria=promotion,
    )


def build_deepar_search_space(
    diagnostics: dict[str, Any],
    cfg: SearchDesignConfig,
) -> ModelSearchSpace:
    summary = _extract_model_summary_map(diagnostics).get("DeepAR", {})
    coverage = float(summary.get("coverage_80", 0.0))
    collapse = bool(summary.get("interval_collapse_warning", coverage <= 0.05))

    collapse_text = (
        "interval collapse warning present"
        if collapse
        else "severe under-coverage persists"
    )
    focus = (
        "Recover interval quality and calibration before ranking point accuracy; "
        f"observed coverage_80={coverage:.4f} ({collapse_text})."
    )

    params = [
        {
            "name": "distribution_loss",
            "type": "categorical",
            "values": ["StudentT", "Normal", "Poisson"],
            "priority_rank": 1,
            "reason": "Distributional choice directly drives interval behavior.",
            "expected_effect": "Reduce interval collapse risk.",
            "cost_level": "low",
        },
        {
            "name": "learning_rate",
            "type": "float_log",
            "min": 1e-4,
            "max": 2e-3,
            "priority_rank": 2,
            "reason": "Stability-sensitive for autoregressive likelihood training.",
            "expected_effect": "Improved convergence and quantile realism.",
            "cost_level": "low",
        },
        {
            "name": "lstm_hidden_size",
            "type": "int",
            "min": 32,
            "max": 128,
            "priority_rank": 3,
            "reason": "Capacity may be insufficient for uncertainty structure.",
            "expected_effect": "Better variance modeling.",
            "cost_level": "medium",
        },
        {
            "name": "lstm_dropout",
            "type": "float",
            "min": 0.05,
            "max": 0.35,
            "priority_rank": 4,
            "reason": "Regularization can reduce overconfident intervals.",
            "expected_effect": "Wider and better-calibrated intervals.",
            "cost_level": "low",
        },
        {
            "name": "max_steps",
            "type": "int",
            "min": 40,
            "max": 220,
            "priority_rank": 5,
            "reason": "Current run may be undertrained for likelihood calibration.",
            "expected_effect": "Potentially improves both calibration and MAE.",
            "cost_level": "medium",
        },
        {
            "name": "input_size_hours",
            "type": "int",
            "min": 96,
            "max": 336,
            "priority_rank": 6,
            "reason": "Longer history can stabilize latent state uncertainty.",
            "expected_effect": "Reduced horizon-level collapse variability.",
            "cost_level": "medium",
        },
    ]

    promotion = [
        (
            f"{cfg.secondary_metric} in "
            f"[{cfg.promotion_coverage_min:.2f}, {cfg.promotion_coverage_max:.2f}]"
        ),
        "No interval collapse warning",
        (
            "No more than "
            f"{cfg.promotion_mae_regression_limit:.2f} relative MAE regression "
            "vs current DeepAR baseline"
        ),
    ]

    stop = [
        "Stop first pass after max_trials_first_pass complete",
        "Stop early if 3 trials still show collapse warning",
        "Stop if runtime exceeds timeout_minutes_first_pass",
    ]

    return ModelSearchSpace(
        model="DeepAR",
        objective_focus=focus,
        parameters=params,
        recommended_first_search_size=cfg.max_trials_first_pass,
        estimated_cost_level="medium",
        stop_criteria=stop,
        promotion_criteria=promotion,
    )


def recommend_search_strategy(
    diagnostics: dict[str, Any],
    cfg: SearchDesignConfig,
) -> dict[str, Any]:
    cfg.validate()

    tft_space = build_tft_search_space(diagnostics, cfg)
    deepar_space = build_deepar_search_space(diagnostics, cfg)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "zone": str(diagnostics.get("zone", "AEP")),
        "inputs": {
            "target_coverage": diagnostics.get("target_coverage", 0.80),
            "models": diagnostics.get("models", ["TFT", "DeepAR"]),
        },
        "strategy": {
            "approach": "focused_small_search",
            "do_not_run_full_optuna_yet": True,
            "first_pass_trials": cfg.max_trials_first_pass,
            "second_pass_trials": cfg.max_trials_second_pass,
            "timeout_minutes_first_pass": cfg.timeout_minutes_first_pass,
            "primary_metric": cfg.primary_metric,
            "secondary_metric": cfg.secondary_metric,
        },
        "spaces": {
            "TFT": tft_space.__dict__,
            "DeepAR": deepar_space.__dict__,
        },
        "promotion_gate": {
            "coverage_min": cfg.promotion_coverage_min,
            "coverage_max": cfg.promotion_coverage_max,
            "mae_regression_limit": cfg.promotion_mae_regression_limit,
        },
    }


def write_search_design(
    strategy: dict[str, Any],
    *,
    zone: str,
    report_root: Path = Path("data/cache/reports"),
) -> tuple[Path, Path]:
    root = _resolve_path(report_root)
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    zone_slug = zone.lower()
    json_path = root / f"{zone_slug}_focused_search_design_{stamp}.json"
    md_path = root / f"{zone_slug}_focused_search_design_{stamp}.md"

    json_path.write_text(json.dumps(strategy, indent=2), encoding="utf-8")

    tft = strategy.get("spaces", {}).get("TFT", {})
    deepar = strategy.get("spaces", {}).get("DeepAR", {})

    def _params_md(space: dict[str, Any]) -> str:
        rows = space.get("parameters", [])
        if not isinstance(rows, list) or not rows:
            return "(no parameters)"
        header = (
            "| name | type | priority_rank | min | max | values | reason | "
            "expected_effect | cost_level |"
        )
        sep = "|---|---|---:|---:|---:|---|---|---|---|"
        body = []
        for row in rows:
            body.append(
                (
                    "| {name} | {type} | {priority_rank} | {minv} | {maxv} | {values} | "
                    "{reason} | {effect} | {cost} |"
                ).format(
                    name=row.get("name", ""),
                    type=row.get("type", ""),
                    priority_rank=row.get("priority_rank", ""),
                    minv=row.get("min", ""),
                    maxv=row.get("max", ""),
                    values=row.get("values", ""),
                    reason=row.get("reason", ""),
                    effect=row.get("expected_effect", ""),
                    cost=row.get("cost_level", ""),
                )
            )
        return "\n".join([header, sep, *body])

    lines = [
        f"# Focused search design — {zone}",
        "",
        f"Generated at: {strategy.get('generated_at')}",
        f"Approach: {strategy.get('strategy', {}).get('approach')}",
        f"First pass trials: {strategy.get('strategy', {}).get('first_pass_trials')}",
        f"Second pass trials: {strategy.get('strategy', {}).get('second_pass_trials')}",
        f"Primary metric: {strategy.get('strategy', {}).get('primary_metric')}",
        f"Secondary metric: {strategy.get('strategy', {}).get('secondary_metric')}",
        "",
        "## TFT",
        f"Objective focus: {tft.get('objective_focus', '')}",
        _params_md(tft),
        "",
        "## DeepAR",
        f"Objective focus: {deepar.get('objective_focus', '')}",
        _params_md(deepar),
        "",
        "## Promotion gate",
        json.dumps(strategy.get("promotion_gate", {}), indent=2),
        "",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
