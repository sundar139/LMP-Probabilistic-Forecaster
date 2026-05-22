"""Tests for real PJM CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lmp_forecaster.cli import app


def test_inspect_pjm_api_dry_run_without_key_does_not_write() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect-pjm-api",
            "--zone",
            "AEP",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-02",
        ],
    )
    assert result.exit_code == 0
    assert "PJM_API_KEY" in result.stdout


def test_pull_real_pjm_lmp_dry_run_no_write() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "pull-real-pjm-lmp",
            "--zone",
            "AEP",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
        ],
    )
    assert result.exit_code == 0
    assert "Dry-run only" in result.stdout


def test_pull_real_pjm_lmp_write_without_key_fails_clearly() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "pull-real-pjm-lmp",
            "--zone",
            "AEP",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
            "--write",
        ],
    )
    assert result.exit_code != 0
    assert "PJM_API_KEY" in result.stdout


def test_generated_paths_and_env_are_ignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "data/raw/**" in gitignore
    assert "data/cache/**" in gitignore


def test_no_forbidden_word_in_repo_names() -> None:
    blocked = {".git", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
    names = [
        p.name.lower()
        for p in Path(".").resolve().rglob("*")
        if not any(part in blocked for part in p.parts)
    ]
    assert all("phase" not in n for n in names)
