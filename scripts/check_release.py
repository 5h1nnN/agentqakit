"""Fail closed when a source tree or built distribution is not release-ready."""

import argparse
import os
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--require-tag", action="store_true")
    return parser


def _distribution_members(path: Path) -> set[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return set(archive.getnames())
    return set()


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    version = project["version"]
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.M):
        raise SystemExit(f"CHANGELOG.md has no dated {version} release entry")
    if args.dist is None and not (root / "agentqa" / "_static" / "dist" / "index.html").is_file():
        raise SystemExit("bundled Web assets are missing; run scripts/sync_web_assets.py")

    if args.require_tag:
        expected = f"v{version}"
        if os.getenv("GITHUB_REF_TYPE") != "tag" or os.getenv("GITHUB_REF_NAME") != expected:
            raise SystemExit(f"publishing requires workflow ref tag {expected}")

    if args.dist is not None:
        artifacts = sorted(args.dist.glob("*"))
        wheels = [path for path in artifacts if path.suffix == ".whl"]
        sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
        if len(wheels) != 1 or len(sdists) != 1:
            raise SystemExit("dist must contain exactly one wheel and one sdist")
        for artifact in [*wheels, *sdists]:
            members = _distribution_members(artifact)
            if not any(name.endswith("agentqa/_static/dist/index.html") for name in members):
                raise SystemExit(f"{artifact.name} does not contain the Web UI")
    print(f"AgentQAKit {version} release checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
