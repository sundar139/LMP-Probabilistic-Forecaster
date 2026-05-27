"""Portable focused-tuning package planning and manifest helpers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.tuning.tuning_runner import load_tuning_config


@dataclass(frozen=True)
class TuningPackageConfig:
    zone: str
    resource_profile: str
    models: tuple[str, ...]
    max_trials: int
    folds: int
    max_steps_cap: int
    panel_path: Path
    baseline_metrics_path: Path
    search_design_path: Path


@dataclass(frozen=True)
class TuningPackageManifest:
    generated_at: str
    zone: str
    resource_profile: str
    repo_commit: str
    repo_branch: str
    python_requirement: str
    uv_commands: list[str]
    tuning_command: str
    import_command_template: str
    expected_inputs: dict[str, str]
    generated_outputs: dict[str, str]
    promotion_gate: dict[str, float | bool]
    hardware_assumptions: dict[str, str]
    required_repo_files: list[str]
    required_config_files: list[str]
    resume_instructions: list[str]
    cleanup_instructions: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Portable focused tuning package — {self.zone}",
            "",
            f"Generated at: {self.generated_at}",
            f"Resource profile: {self.resource_profile}",
            f"Repo commit: {self.repo_commit}",
            f"Repo branch: {self.repo_branch}",
            f"Python requirement: {self.python_requirement}",
            "",
            "## Commands",
        ]
        lines.extend([f"- {cmd}" for cmd in self.uv_commands])
        lines.extend(
            [
                "",
                "## Run tuning elsewhere",
                f"- {self.tuning_command}",
                "",
                "## Import command template",
                f"- {self.import_command_template}",
                "",
                "## Expected inputs",
            ]
        )
        lines.extend([f"- {k}: {v}" for k, v in self.expected_inputs.items()])
        lines.extend(["", "## Generated outputs"])
        lines.extend([f"- {k}: {v}" for k, v in self.generated_outputs.items()])
        lines.extend(["", "## Promotion gate"])
        lines.extend([f"- {k}: {v}" for k, v in self.promotion_gate.items()])
        lines.extend(["", "## Hardware assumptions"])
        lines.extend([f"- {k}: {v}" for k, v in self.hardware_assumptions.items()])
        lines.extend(["", "## Required repo files"])
        lines.extend([f"- {p}" for p in self.required_repo_files])
        lines.extend(["", "## Required config files"])
        lines.extend([f"- {p}" for p in self.required_config_files])
        lines.extend(["", "## Resume instructions"])
        lines.extend([f"- {item}" for item in self.resume_instructions])
        lines.extend(["", "## Cleanup instructions"])
        lines.extend([f"- {item}" for item in self.cleanup_instructions])
        lines.extend(["", "## Notes"])
        lines.extend([f"- {item}" for item in self.notes])
        return "\n".join(lines) + "\n"


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            capture_output=True,
            cwd=None if cwd is None else str(cwd),
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _to_rel(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def collect_required_repo_files(project_root: Path | None = None) -> list[str]:
    root = project_root or get_project_paths().root
    candidates = [
        root / "pyproject.toml",
        root / "uv.lock",
        root / "README.md",
        root / "conf" / "tuning.yaml",
        root / "src" / "lmp_forecaster" / "cli.py",
        root / "src" / "lmp_forecaster" / "tuning" / "tuning_runner.py",
        root / "src" / "lmp_forecaster" / "tuning" / "promotion.py",
        root / "src" / "lmp_forecaster" / "tuning" / "package.py",
        root / "src" / "lmp_forecaster" / "tuning" / "result_import.py",
        root / "scripts" / "export_tuning_package.ps1",
        root / "scripts" / "import_tuning_results.ps1",
        root / "docs" / "focused_tuning.md",
        root / "docs" / "portable_tuning_workflow.md",
    ]
    return [_to_rel(path, root) for path in candidates if path.exists()]


def collect_required_config_files(project_root: Path | None = None) -> list[str]:
    root = project_root or get_project_paths().root
    candidates = [
        root / "conf" / "tuning.yaml",
        root / ".env.example",
    ]
    return [_to_rel(path, root) for path in candidates if path.exists()]


def collect_required_command_plan(cfg: TuningPackageConfig) -> list[str]:
    model_list = ",".join(cfg.models)
    tuning_cmd = (
        "uv run python -m lmp_forecaster.cli run-focused-tuning "
        f"--zone {cfg.zone} "
        f"--resource-profile {cfg.resource_profile} "
        f"--models {model_list} "
        f"--max-trials {cfg.max_trials} "
        f"--folds {cfg.folds} "
        f"--max-steps-cap {cfg.max_steps_cap} --write"
    )
    return [
        "uv sync --frozen",
        "uv run ruff check .",
        "uv run mypy src",
        "uv run pytest -q",
        tuning_cmd,
    ]


def create_tuning_package(
    cfg: TuningPackageConfig,
    project_root: Path | None = None,
) -> TuningPackageManifest:
    paths = get_project_paths(project_root)
    run_cfg = load_tuning_config(paths.root / "conf" / "tuning.yaml")

    commands = collect_required_command_plan(cfg)
    model_list = ",".join(cfg.models)
    tuning_command = commands[-1]
    import_template = (
        "uv run python -m lmp_forecaster.cli import-tuning-results "
        f"--zone {cfg.zone} "
        f"--ranked-candidates-path data/cache/tuning/{cfg.zone.lower()}"
        "_focused_tuning_ranked_<timestamp>.csv "
        f"--summary-path data/cache/reports/{cfg.zone.lower()}"
        "_focused_tuning_summary_<timestamp>.json --write"
    )

    expected_inputs = {
        "panel_path": cfg.panel_path.as_posix(),
        "baseline_metrics_path": cfg.baseline_metrics_path.as_posix(),
        "search_design_path": cfg.search_design_path.as_posix(),
        "required_data_note": (
            "Bring private panel/backtest artifacts to external runner separately. "
            "Do not commit private data into repo."
        ),
    }

    generated_outputs = {
        "package_manifest_json": (
            "data/cache/tuning_packages/<zone>_tuning_package_<timestamp>.json"
        ),
        "package_manifest_markdown": (
            "data/cache/tuning_packages/<zone>_tuning_package_<timestamp>.md"
        ),
        "tuning_trials_csv": (
            "data/cache/tuning/<zone>_focused_tuning_trials_<timestamp>.csv"
        ),
        "tuning_ranked_csv": (
            "data/cache/tuning/<zone>_focused_tuning_ranked_<timestamp>.csv"
        ),
        "tuning_summary_json": (
            "data/cache/reports/<zone>_focused_tuning_summary_<timestamp>.json"
        ),
        "tuning_summary_markdown": (
            "data/cache/reports/<zone>_focused_tuning_summary_<timestamp>.md"
        ),
        "tuning_manifest_json": (
            "artifacts/tuning/<zone>_focused_tuning_manifest_<timestamp>.json"
        ),
    }

    profile = run_cfg.resource_profiles.get(cfg.resource_profile, run_cfg.profile)

    return TuningPackageManifest(
        generated_at=datetime.now(UTC).isoformat(),
        zone=cfg.zone,
        resource_profile=cfg.resource_profile,
        repo_commit=_run_git(["rev-parse", "HEAD"], cwd=paths.root),
        repo_branch=_run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=paths.root),
        python_requirement=">=3.12,<3.13",
        uv_commands=commands,
        tuning_command=tuning_command,
        import_command_template=import_template,
        expected_inputs=expected_inputs,
        generated_outputs=generated_outputs,
        promotion_gate={
            "coverage_min": run_cfg.coverage_min,
            "coverage_max": run_cfg.coverage_max,
            "mae_regression_limit": run_cfg.mae_regression_limit,
            "allow_deepar_if_interval_collapse": run_cfg.allow_deepar_if_interval_collapse,
        },
        hardware_assumptions={
            "profile_description": profile.description,
            "max_trials": str(profile.max_trials),
            "folds": str(profile.folds),
            "max_steps_cap": str(profile.max_steps_cap),
            "batch_size": str(profile.batch_size),
        },
        required_repo_files=collect_required_repo_files(paths.root),
        required_config_files=collect_required_config_files(paths.root),
        resume_instructions=[
            "If external run is interrupted, re-run the same command with identical inputs.",
            "Keep generated outputs under data/cache/tuning and data/cache/reports.",
            "Import ranked candidates back locally for promotion recompute.",
        ],
        cleanup_instructions=[
            "Delete temporary checkpoint/log directories after cloud run.",
            "Retain only ranked CSV + summary JSON needed for local import validation.",
            "Never commit private data or secrets from external environment.",
        ],
        notes=[
            f"Models planned: {model_list}",
            "Imported promotion labels are advisory only; local recompute is authoritative.",
        ],
    )


def validate_tuning_package(manifest: TuningPackageManifest) -> list[str]:
    issues: list[str] = []
    payload = json.dumps(manifest.to_dict(), sort_keys=True).lower()

    if "pjm_api_key" in payload or "api_key" in payload:
        issues.append("Manifest must not include API keys.")

    if "/.env" in payload or "\\.env" in payload or '".env"' in payload:
        issues.append("Manifest must not include .env paths.")

    if "c:/users/" in payload or "c:\\users\\" in payload:
        issues.append("Manifest should avoid local absolute user paths.")

    allowed_output_roots = (
        "data/cache/tuning_packages/",
        "data/cache/tuning/",
        "data/cache/reports/",
        "artifacts/tuning/",
    )
    for key, value in manifest.generated_outputs.items():
        normalized = value.replace("\\", "/")
        if not normalized.startswith(allowed_output_roots):
            issues.append(f"Output path for {key} is outside ignored roots: {value}")

    return issues


def write_tuning_package_manifest(
    manifest: TuningPackageManifest,
    output_root: Path | None = None,
) -> dict[str, Path]:
    root = output_root or Path("data/cache/tuning_packages")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    zone = manifest.zone.lower()
    json_path = root / f"{zone}_tuning_package_{stamp}.json"
    md_path = root / f"{zone}_tuning_package_{stamp}.md"

    json_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(manifest.to_markdown(), encoding="utf-8")

    return {"json": json_path, "markdown": md_path}
