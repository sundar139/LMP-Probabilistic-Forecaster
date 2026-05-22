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


def test_cli_tracking_flag_visible_in_dry_run() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train-single-zone-baselines",
            "--zone",
            "AEP",
            "--enable-tracking",
            "--tracking-uri",
            "file:./mlruns",
            "--experiment-name",
            "exp_test",
            "--run-name",
            "run_test",
        ],
    )
    assert result.exit_code == 0
    assert "tracking_enabled=True" in result.stdout


def test_training_dry_run_does_not_create_mlruns(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        Path("conf").mkdir(parents=True, exist_ok=True)
        Path("conf/zones.yaml").write_text(
            "zones:\n"
            "  - zone: AEP\n"
            "    display_name: AEP\n"
            "    latitude: 39.96\n"
            "    longitude: -82.99\n"
            "    region_cluster: west\n"
            "    type: ZONE\n",
            encoding="utf-8",
        )
        Path("conf/training.yaml").write_text(
            "training:\n"
            "  horizon_hours: 24\n"
            "  input_size_hours: 168\n"
            "  quantiles: [0.1, 0.5, 0.9]\n"
            "  interval_level: 80\n"
            "  validation_hours: 72\n"
            "  test_hours: 72\n"
            "  seed: 42\n"
            "  max_steps_smoke: 30\n"
            "  max_steps_real_candidate: 200\n"
            "  batch_size: 32\n"
            "  num_workers: 0\n"
            "  accelerator: auto\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, ["train-single-zone-baselines", "--zone", "AEP"])
        assert result.exit_code == 0
        assert not Path("mlruns").exists()


def test_generated_paths_ignored_by_git() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/cache/**" in gitignore
    assert "artifacts/" in gitignore
    assert "mlruns/" in gitignore


def test_no_forbidden_word_in_repo_names() -> None:
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
        for p in Path(".").resolve().rglob("*")
        if not any(part in blocked for part in p.parts)
    ]
    forbidden = "ph" + "ase"
    assert all(forbidden not in n for n in names)
