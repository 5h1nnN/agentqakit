"""Atomic checksummed persistence for comparison and gate reports."""

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from agentqa.specs.hashing import canonical_json_bytes, sha256_digest


def _replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_report(path: str | Path, report: BaseModel) -> Path:
    """Atomically write canonical JSON plus a SHA-256 sidecar."""
    destination = Path(path).resolve()
    payload = canonical_json_bytes(report.model_dump(mode="json")) + b"\n"
    _replace(destination, payload)
    _replace(
        destination.with_name(f"{destination.name}.sha256"),
        f"{sha256_digest(payload)}\n".encode(),
    )
    return destination
