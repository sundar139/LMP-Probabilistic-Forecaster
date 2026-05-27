"""Policy tests for external agent memory setup and repo ignore safeguards."""

from __future__ import annotations

from pathlib import Path


def test_project_memory_file_is_outside_repo() -> None:
    repo_root = Path(".").resolve()
    memory_file = Path(
        "C:/Users/rohit/Documents/Agent Memory/projects/lmp-probabilistic-forecaster.md"
    ).resolve()

    assert memory_file.exists()
    assert repo_root not in memory_file.parents


def test_repo_gitignore_includes_memory_mirror_patterns() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    for required in [
        ".agent-memory/",
        "AGENT_MEMORY*.md",
        ".hermes-memory*.md",
        ".codex-memory*.md",
    ]:
        assert required in gitignore


def test_no_forbidden_word_in_file_or_folder_names() -> None:
    blocked = {
        ".git",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "data",
        "artifacts",
        "lightning_logs",
        "mlruns",
        "checkpoints",
    }
    forbidden = "ph" + "ase"
    for path in Path(".").resolve().rglob("*"):
        if any(part in blocked for part in path.parts):
            continue
        assert forbidden not in path.name.lower()
