"""Copy the verified Vite build into the Python package before distribution."""

import shutil
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "apps" / "web" / "dist"
    destination = root / "agentqa" / "_static" / "dist"
    if not (source / "index.html").is_file():
        raise SystemExit("Web build missing: run npm run build in apps/web first")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    print(f"Copied Web assets to {destination.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
