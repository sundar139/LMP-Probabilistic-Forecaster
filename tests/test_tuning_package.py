"""Portable tuning package manifest unit tests."""

from __future__ import annotations

import json
from pathlib import Path

from lmp_forecaster.tuning.package import (
    TuningPackageConfig,
    collect_required_command_plan,
    create_tuning_package,
    validate_tuning_package,
    write_tuning_package_manifest,
)


def _write_tuning_conf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "tuning:\n"
        "  zone: AEP\n"
        "  models: [TFT, DeepAR]\n"
        "  max_trials_first_pass: 12\n"
        "  folds_for_full_first_pass: 2\n"
        "  horizon_hours: 24\n"
        "  max_steps_cap: 60\n"
        "  target_coverage: 0.80\n"
        "  coverage_min: 0.70\n"
        "  coverage_max: 0.90\n"
        "  mae_regression_limit: 0.15\n"
        "resource_profiles:\n"
        "  local_safe:\n"
        "    description: Local laptop-safe profile for 8GB VRAM / 16GB RAM\n"
        "    max_trials: 2\n"
        "    folds: 1\n"
        "    max_steps_cap: 3\n"
        "    batch_size: 4\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: false\n"
        "  cloud_16gb:\n"
        "    description: Moderate cloud GPU profile, intended for 16GB VRAM\n"
        "    max_trials: 12\n"
        "    folds: 2\n"
        "    max_steps_cap: 50\n"
        "    batch_size: 8\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: true\n"
        "  cloud_24gb:\n"
        "    description: Larger cloud GPU profile, intended for 24GB+ VRAM\n"
        "    max_trials: 30\n"
        "    folds: 3\n"
        "    max_steps_cap: 100\n"
        "    batch_size: 16\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: true\n",
        encoding="utf-8",
    )


def _setup_repo_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0.0.0'\nrequires-python='>=3.12,<3.13'\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# temp\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("PJM_API_KEY=\n", encoding="utf-8")
    _write_tuning_conf(tmp_path / "conf/tuning.yaml")


def _package_cfg() -> TuningPackageConfig:
    return TuningPackageConfig(
        zone="AEP",
        resource_profile="cloud_16gb",
        models=("TFT", "DeepAR"),
        max_trials=12,
        folds=2,
        max_steps_cap=50,
        panel_path=Path("data/processed/panel/single_zone/AEP_panel.parquet"),
        baseline_metrics_path=Path("data/cache/backtests/aep_rolling_backtest_aggregate_metrics.csv"),
        search_design_path=Path("data/cache/reports/aep_focused_search_design.json"),
    )


def test_cloud_16gb_package_plan_includes_cloud_profile_without_training_execution(
    tmp_path: Path,
) -> None:
    _setup_repo_root(tmp_path)
    cfg = _package_cfg()

    manifest = create_tuning_package(cfg, project_root=tmp_path)

    assert manifest.resource_profile == "cloud_16gb"
    assert "--resource-profile cloud_16gb" in manifest.tuning_command
    assert "run-focused-tuning" in manifest.tuning_command
    assert not list((tmp_path / "data/cache/tuning").glob("*.csv"))


def test_package_manifest_excludes_env_and_api_keys(tmp_path: Path) -> None:
    _setup_repo_root(tmp_path)
    manifest = create_tuning_package(_package_cfg(), project_root=tmp_path)

    issues = validate_tuning_package(manifest)
    payload = json.dumps(manifest.to_dict(), sort_keys=True).lower()

    assert issues == []
    assert "pjm_api_key" not in payload
    assert '".env"' not in payload


def test_package_manifest_includes_commit_hash_and_uv_commands(tmp_path: Path) -> None:
    _setup_repo_root(tmp_path)
    manifest = create_tuning_package(_package_cfg(), project_root=tmp_path)

    assert manifest.repo_commit
    assert manifest.repo_branch
    assert manifest.uv_commands
    assert "uv sync --frozen" in manifest.uv_commands
    assert "uv run ruff check ." in manifest.uv_commands
    assert "uv run mypy src" in manifest.uv_commands
    assert "uv run pytest -q" in manifest.uv_commands


def test_package_manifest_includes_resource_profile_command_plan_and_promotion_gates(
    tmp_path: Path,
) -> None:
    _setup_repo_root(tmp_path)
    cfg = _package_cfg()

    manifest = create_tuning_package(cfg, project_root=tmp_path)
    command_plan = collect_required_command_plan(cfg)

    assert manifest.hardware_assumptions["profile_description"]
    assert manifest.hardware_assumptions["max_trials"] == "12"
    assert manifest.hardware_assumptions["folds"] == "2"
    assert manifest.promotion_gate["coverage_min"] == 0.7
    assert manifest.promotion_gate["coverage_max"] == 0.9
    assert manifest.promotion_gate["mae_regression_limit"] == 0.15
    assert command_plan[-1] == manifest.tuning_command


def test_write_manifest_writes_only_under_ignored_output_root(tmp_path: Path) -> None:
    _setup_repo_root(tmp_path)
    manifest = create_tuning_package(_package_cfg(), project_root=tmp_path)

    output_paths = write_tuning_package_manifest(
        manifest,
        output_root=tmp_path / "data/cache/tuning_packages",
    )

    for path in output_paths.values():
        rendered = str(path).replace("\\", "/")
        assert "/data/cache/tuning_packages/" in rendered
        assert path.exists()
