"""Release-time scanning for secrets and over-broad workbench exchange artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_SCAN_BYTES = 16 * 1024 * 1024
_SECRET_PATTERNS = (
    ("OPENAI_STYLE_KEY", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    (
        "AUTHORIZATION_HEADER",
        re.compile(r"(?i)\bauthorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._~-]{12,}"),
    ),
    (
        "PRIVATE_KEY",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
)
_BOUNDED_EXCHANGE_NAMES = frozenset({"handoff-bundle.json", "workflow-report.json"})
_FORBIDDEN_EXCHANGE_KEYS = frozenset(
    {"api_key", "authorization", "code_evidence", "content", "prompt", "system_prompt"}
)


@dataclass(frozen=True, slots=True)
class LeakFinding:
    path: str
    code: str
    location: str


def _walk_keys(value: Any, location: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}"
            yield str(key), child
            yield from _walk_keys(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_keys(item, f"{location}[{index}]")


def scan_artifact_leaks(roots: Iterable[str | Path]) -> list[LeakFinding]:
    """Scan JSON exchange artifacts without printing any matched secret value."""
    findings: list[LeakFinding] = []
    seen: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root)
        paths = [root] if root.is_file() else root.rglob("*") if root.is_dir() else []
        for path in paths:
            if path in seen or not path.is_file() or path.is_symlink():
                continue
            seen.add(path)
            if path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml"}:
                continue
            try:
                if path.stat().st_size > _MAX_SCAN_BYTES:
                    findings.append(LeakFinding(str(path), "SCAN_SIZE_LIMIT", "$"))
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                findings.append(LeakFinding(str(path), "SCAN_READ_ERROR", "$"))
                continue
            for code, pattern in _SECRET_PATTERNS:
                if pattern.search(text):
                    findings.append(LeakFinding(str(path), code, "$"))
            if path.name not in _BOUNDED_EXCHANGE_NAMES:
                continue
            try:
                document = json.loads(text)
            except json.JSONDecodeError:
                findings.append(LeakFinding(str(path), "INVALID_EXCHANGE_JSON", "$"))
                continue
            for key, location in _walk_keys(document):
                if key.lower() in _FORBIDDEN_EXCHANGE_KEYS:
                    findings.append(LeakFinding(str(path), "OVER_BROAD_EXCHANGE_FIELD", location))
    return sorted(findings, key=lambda item: (item.path, item.code, item.location))


__all__ = ["LeakFinding", "scan_artifact_leaks"]
