"""Tests for training CLI behavior."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lmp_forecaster.cli import app


def test_cli_training_dry_run_does_not_write() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["train-single-zone-baselines", "--zone", "AEP"])
    assert result.exit_code == 0
    assert "Dry-run only" in result.stdout


def test_cli_synthetic_warning_in_dry_run() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train-single-zone-baselines",
            "--zone",
            "AEP",
            "--allow-synthetic-panel",
            "--build-panel-if-missing",
        ],
    )
    assert result.exit_code == 0
    assert "Synthetic panel option enabled" in result.stdout


def test_generated_paths_ignored_by_git() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/cache/**" in gitignore
    assert "artifacts/" in gitignore


def test_no_phase_in_repo_names() -> None:
    blocked = {".git", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
    names = [
        p.name.lower()
        for p in Path(".").resolve().rglob("*")
        if not any(part in blocked for part in p.parts)
    ]
    assert all("phase" not in n for n in names)
