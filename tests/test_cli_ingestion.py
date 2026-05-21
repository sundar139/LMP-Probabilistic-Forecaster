"""Tests for ingestion CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from lmp_forecaster.cli import app


@dataclass(frozen=True)
class _StubResult:
    output_path: Path
    normalized: pd.DataFrame
    request_url: str
    request_params: dict[str, str | float]


def _write_minimal_repo_scaffold(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    (tmp_path / "conf").mkdir(parents=True, exist_ok=True)
    zones_yaml = (
        "zones:\n"
        "  - zone: AEP\n"
        "    display_name: AEP\n"
        "    latitude: 39.96\n"
        "    longitude: -82.99\n"
        "    region_cluster: west\n"
        "    type: ZONE\n"
    )
    (tmp_path / "conf" / "zones.yaml").write_text(zones_yaml, encoding="utf-8")


def test_cli_dry_run_does_not_write(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()

    monkeypatch.chdir(tmp_path)
    _write_minimal_repo_scaffold(tmp_path)

    result = runner.invoke(app, ["pull-weather-smoke", "--zone", "AEP"])

    assert result.exit_code == 0
    assert "Dry-run only" in result.stdout
    cache_dir = tmp_path / "data" / "cache"
    assert not cache_dir.exists() or not any(cache_dir.rglob("*.parquet"))


def test_cli_write_mode_writes_under_cache_only(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    _write_minimal_repo_scaffold(tmp_path)

    def fake_pull(cfg, *, write: bool, http_config=None):  # type: ignore[no-untyped-def]
        assert write is True
        path = tmp_path / "data" / "cache" / "weather" / "openmeteo" / "stub.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [1]}).to_parquet(path, index=False)
        return _StubResult(
            output_path=path,
            normalized=pd.DataFrame({"x": [1]}),
            request_url="https://example.test",
            request_params={"latitude": cfg.latitude, "longitude": cfg.longitude},
        )

    monkeypatch.setattr("lmp_forecaster.cli.pull_historical_weather_smoke", fake_pull)

    result = runner.invoke(app, ["pull-weather-smoke", "--zone", "AEP", "--write"])

    assert result.exit_code == 0
    assert "Wrote rows: 1" in result.stdout
    parquet_files = list((tmp_path / "data").rglob("*.parquet"))
    assert len(parquet_files) == 1
    assert "data/cache" in str(parquet_files[0]).replace("\\", "/")
    assert "data/raw" not in str(parquet_files[0]).replace("\\", "/")


def test_cli_has_required_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "list-sources" in result.stdout
    assert "list-zones" in result.stdout
    assert "pull-pjm-smoke" in result.stdout
    assert "pull-weather-smoke" in result.stdout
    assert "pull-historical-forecast-smoke" in result.stdout


def test_no_phase_in_repo_names() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    blocked = {
        ".git",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
    names = [
        p.name.lower()
        for p in repo_root.rglob("*")
        if not any(part in blocked for part in p.parts)
    ]
    assert all("phase" not in name for name in names)


def test_generated_outputs_ignored_by_git() -> None:
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw/**" in gitignore
    assert "data/cache/**" in gitignore
