"""Fail CI when persisted workbench artifacts expose secrets or raw code evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentqa.security import scan_artifact_leaks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or [Path(".agentqa/artifacts"), Path(".agentqa/mcp")]
    findings = scan_artifact_leaks(paths)
    for finding in findings:
        print(f"{finding.path}: {finding.code} at {finding.location}")
    if findings:
        print(f"Artifact leak scan failed with {len(findings)} finding(s).")
        return 1
    print(f"Artifact leak scan passed for {len(paths)} root(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
