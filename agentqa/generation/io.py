"""Safe, checksummed persistence for generation artifacts."""

import os
import tempfile
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from agentqa.application.generation import GenerationError
from agentqa.models import GenerationPlan, GenerationReport
from agentqa.specs import normalize_spec
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest

_Model = TypeVar("_Model", bound=BaseModel)
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


def ensure_workspace_path(path: Path, workspace: Path) -> Path:
    """Resolve one output path and reject escapes from the selected workspace."""
    root = workspace.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise GenerationError(
            "generation output must remain inside the workspace",
            code="GENERATION_OUTPUT_PATH_ERROR",
        ) from None
    return resolved


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _write_checksummed(path: Path, payload: bytes) -> None:
    if path.exists() or path.with_name(f"{path.name}.sha256").exists():
        raise GenerationError(
            f"generation output already exists: {path.name}",
            code="GENERATION_OUTPUT_EXISTS",
        )
    _atomic_write(path, payload)
    _atomic_write(
        path.with_name(f"{path.name}.sha256"),
        f"{sha256_digest(payload)}\n".encode(),
    )


def write_generation_plan(plan: GenerationPlan, path: Path, *, workspace: Path) -> Path:
    """Write one immutable, checksummed generation plan."""
    output = ensure_workspace_path(path, workspace)
    payload = canonical_json_bytes(plan) + b"\n"
    _write_checksummed(output, payload)
    return output


def _read_checksummed_model(path: Path, model: type[_Model]) -> _Model:
    try:
        payload = path.read_bytes()
        expected = path.with_name(f"{path.name}.sha256").read_text(encoding="utf-8").strip()
    except OSError:
        raise GenerationError(
            "generation artifact or checksum could not be read",
            code="GENERATION_ARTIFACT_READ_ERROR",
        ) from None
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise GenerationError(
            "generation artifact exceeds the size limit",
            code="GENERATION_ARTIFACT_TOO_LARGE",
        )
    if expected != sha256_digest(payload):
        raise GenerationError(
            "generation artifact checksum mismatch",
            code="GENERATION_ARTIFACT_CHECKSUM_ERROR",
        )
    try:
        return model.model_validate_json(payload)
    except ValidationError:
        raise GenerationError(
            "generation artifact does not match its schema",
            code="GENERATION_ARTIFACT_SCHEMA_ERROR",
        ) from None


def load_generation_plan(path: Path) -> GenerationPlan:
    """Load and verify one plan produced by :func:`write_generation_plan`."""
    return _read_checksummed_model(path, GenerationPlan)


def write_generation_suite(
    report: GenerationReport,
    directory: Path,
    *,
    workspace: Path,
) -> list[Path]:
    """Write generated YAML specs and their report into a new or empty directory."""
    output = ensure_workspace_path(directory, workspace)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise GenerationError(
            "generated test output directory must be absent or empty",
            code="GENERATION_OUTPUT_NOT_EMPTY",
        )
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case in report.cases:
        path = output / f"{case.case_id}.yaml"
        payload = yaml.safe_dump(
            normalize_spec(case.spec),
            allow_unicode=True,
            sort_keys=False,
        ).encode("utf-8")
        _write_checksummed(path, payload)
        written.append(path)
    report_path = output / "generation-report.json"
    report_payload = canonical_json_bytes(report) + b"\n"
    _write_checksummed(report_path, report_payload)
    written.append(report_path)
    return written


__all__ = [
    "ensure_workspace_path",
    "load_generation_plan",
    "write_generation_plan",
    "write_generation_suite",
]
