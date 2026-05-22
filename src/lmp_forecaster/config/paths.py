"""Filesystem path helpers for the forecasting project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths rooted at repository base."""

    root: Path
    conf: Path
    data: Path
    raw: Path
    cache: Path
    processed: Path
    artifacts: Path
    mlruns: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        """Create a path registry from a repository root."""
        return cls(
            root=root,
            conf=root / "conf",
            data=root / "data",
            raw=root / "data" / "raw",
            cache=root / "data" / "cache",
            processed=root / "data" / "processed",
            artifacts=root / "artifacts",
            mlruns=root / "mlruns",
        )

    def ensure_directories(self) -> None:
        """Ensure writable directories exist locally."""
        for folder in [
            self.conf,
            self.data,
            self.raw,
            self.cache,
            self.processed,
            self.artifacts,
        ]:
            folder.mkdir(parents=True, exist_ok=True)


def discover_project_root(start: Path | None = None) -> Path:
    """Discover project root by locating pyproject.toml in current or parent folders."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root. Expected pyproject.toml in current directory or parents."
    )


def get_project_paths(start: Path | None = None) -> ProjectPaths:
    """Return resolved project paths and ensure base directories exist."""
    paths = ProjectPaths.from_root(discover_project_root(start=start))
    paths.ensure_directories()
    return paths
