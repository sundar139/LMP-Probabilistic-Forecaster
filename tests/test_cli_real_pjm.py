"""Tests for real PJM CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lmp_forecaster.cli import app


def test_inspect_pjm_api_dry_run_without_key_does_not_write(monkeypatch) -> None:
    runner = CliRunner()

    from lmp_forecaster import cli as cli_module

    class DummyCfg:
        api_base_url = "https://api.pjm.com/api/v1"
        api_key = None
        timezone = "America/New_York"

    monkeypatch.setattr(cli_module, "resolve_pjm_api_config", lambda: DummyCfg())

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
    assert "Chunk days: 7" in result.stdout


def test_pull_real_pjm_lmp_write_without_key_fails_clearly(monkeypatch) -> None:
    runner = CliRunner()

    from lmp_forecaster import cli as cli_module

    class DummyCfg:
        api_base_url = "https://api.pjm.com/api/v1"
        api_key = None
        timezone = "America/New_York"
        max_connections_per_minute = 5

    monkeypatch.setattr(cli_module, "resolve_pjm_api_config", lambda: DummyCfg())

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


def test_pull_real_pjm_lmp_dry_run_shows_effective_connection_limit(monkeypatch) -> None:
    runner = CliRunner()

    from lmp_forecaster import cli as cli_module

    class DummyCfg:
        api_base_url = "https://api.pjm.com/api/v1"
        api_key = None
        timezone = "America/New_York"
        max_connections_per_minute = 5

    monkeypatch.setattr(cli_module, "resolve_pjm_api_config", lambda: DummyCfg())

    result = runner.invoke(
        app,
        [
            "pull-real-pjm-lmp",
            "--zone",
            "AEP",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-12-31",
        ],
    )
    assert result.exit_code == 0
    assert "Chunks planned: 53" in result.stdout
    assert "Effective PJM connection limit (/min): 5" in result.stdout


def test_generated_paths_and_env_are_ignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "data/raw/**" in gitignore
    assert "data/cache/**" in gitignore


def test_no_forbidden_word_in_repo_names() -> None:
    blocked = {".git", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
    root = Path(".").resolve()

    for path in root.rglob("*"):
        if any(part in blocked for part in path.parts):
            continue
        forbidden = "ph" + "ase"
        assert forbidden not in path.name.lower()
