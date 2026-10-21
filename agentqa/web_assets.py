"""Locate the React application in a source checkout or installed distribution."""

from importlib.resources import files
from pathlib import Path


def _is_web_build(path: Path) -> bool:
    return path.is_dir() and (path / "index.html").is_file()


def bundled_web_directory() -> Path | None:
    """Return the UI embedded in an installed wheel, when one is available."""
    candidate = Path(str(files("agentqa._static").joinpath("dist")))
    return candidate if _is_web_build(candidate) else None


def resolve_web_directory(
    workspace: Path,
    *,
    explicit: Path | None = None,
) -> Path | None:
    """Prefer an explicit path, then a source build, then packaged Web assets."""
    if explicit is not None:
        return explicit.resolve()
    source_build = workspace.resolve() / "apps" / "web" / "dist"
    if _is_web_build(source_build):
        return source_build
    return bundled_web_directory()


__all__ = ["bundled_web_directory", "resolve_web_directory"]
