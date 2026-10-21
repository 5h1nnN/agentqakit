from pathlib import Path

from pytest import MonkeyPatch

from agentqa import web_assets


def _web_build(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    return path


def test_resolve_web_directory_prefers_source_build(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = _web_build(tmp_path / "apps" / "web" / "dist")
    bundled = _web_build(tmp_path / "bundled")
    monkeypatch.setattr(web_assets, "bundled_web_directory", lambda: bundled)

    assert web_assets.resolve_web_directory(tmp_path) == source


def test_resolve_web_directory_falls_back_to_bundled_build(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    bundled = _web_build(tmp_path / "bundled")
    monkeypatch.setattr(web_assets, "bundled_web_directory", lambda: bundled)

    assert web_assets.resolve_web_directory(tmp_path) == bundled


def test_resolve_web_directory_honors_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom-ui"

    assert web_assets.resolve_web_directory(tmp_path, explicit=explicit) == explicit.resolve()
