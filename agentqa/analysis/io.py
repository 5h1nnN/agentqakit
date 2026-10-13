"""Verified run-evidence loading and checksummed analysis-report persistence."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from agentqa.application.analysis import AnalysisError
from agentqa.comparison import ComparisonError, load_run_artifact
from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    AnalysisReport,
    TestResult,
    TraceRecord,
)
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest

_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class FailureAnalysisInput:
    """Internally consistent immutable inputs for one failed trial."""

    run_id: str
    test_spec: AgentTestSpec
    result: TestResult
    trace: TraceRecord
    manifest: AgentManifest | None


def _checked_document(root: Path, relative: Path, verify: Any) -> dict[str, Any]:
    if relative.is_absolute() or ".." in relative.parts:
        raise AnalysisError("analysis evidence path is unsafe", code="ANALYSIS_PATH_ERROR")
    if not verify(relative):
        raise AnalysisError(
            f"analysis evidence checksum mismatch: {relative.name}",
            code="ANALYSIS_CHECKSUM_ERROR",
        )
    path = root / relative
    try:
        payload = path.read_bytes()
    except OSError:
        raise AnalysisError(
            f"analysis evidence could not be read: {relative.name}",
            code="ANALYSIS_READ_ERROR",
        ) from None
    if len(payload) > _MAX_DOCUMENT_BYTES:
        raise AnalysisError(
            f"analysis evidence is too large: {relative.name}",
            code="ANALYSIS_EVIDENCE_TOO_LARGE",
        )
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeError):
        raise AnalysisError(
            f"analysis evidence is not valid JSON: {relative.name}",
            code="ANALYSIS_FORMAT_ERROR",
        ) from None
    if not isinstance(value, dict):
        raise AnalysisError(
            f"analysis evidence must be an object: {relative.name}",
            code="ANALYSIS_FORMAT_ERROR",
        )
    document = cast(dict[str, Any], value)
    document.pop("schema_version", None)
    return document


def load_failure_analysis_input(
    run_directory: Path,
    trial_id: str,
) -> FailureAnalysisInput:
    """Load one failed trial plus its exact checksummed test, trace, and Manifest."""
    try:
        artifact = load_run_artifact(run_directory)
    except ComparisonError as exc:
        raise AnalysisError(exc.message, code=exc.code.replace("COMPARISON", "ANALYSIS")) from None
    matches = [item for item in artifact.results if item.trial_id == trial_id]
    if len(matches) != 1:
        raise AnalysisError(
            "run must contain exactly one matching trial",
            code="ANALYSIS_TRIAL_SELECTION_ERROR",
        )
    result = matches[0]
    if result.passed:
        raise AnalysisError("selected trial passed", code="ANALYSIS_NOT_FAILED")
    test_matches = [item for item in artifact.tests if item.metadata.name == result.test_case_id]
    if len(test_matches) != 1:
        raise AnalysisError(
            "failed trial does not have one matching AgentTest",
            code="ANALYSIS_TEST_SELECTION_ERROR",
        )
    trace_reference = Path(result.trace_ref or f"traces/{result.trace_id}.json")
    if (
        trace_reference.parent != Path("traces")
        or trace_reference.name != f"{result.trace_id}.json"
    ):
        raise AnalysisError("failed trial trace reference is unsafe", code="ANALYSIS_PATH_ERROR")
    try:
        trace = TraceRecord.model_validate_json(
            json.dumps(
                _checked_document(
                    artifact.root,
                    trace_reference,
                    artifact.reader.verify_checksum,
                )
            )
        )
    except ValidationError:
        raise AnalysisError(
            "failed trial trace does not match TraceRecord",
            code="ANALYSIS_FORMAT_ERROR",
        ) from None

    manifest = None
    manifest_path = artifact.root / "resolved-manifest.json"
    if manifest_path.exists():
        document = _checked_document(
            artifact.root,
            Path("resolved-manifest.json"),
            artifact.reader.verify_checksum,
        )
        try:
            manifest = AgentManifest.model_validate(document["manifest"])
        except (KeyError, ValidationError, TypeError):
            raise AnalysisError(
                "resolved Manifest does not match AgentManifest",
                code="ANALYSIS_FORMAT_ERROR",
            ) from None
    return FailureAnalysisInput(
        run_id=artifact.run_id,
        test_spec=test_matches[0],
        result=result,
        trace=trace,
        manifest=manifest,
    )


def _workspace_path(path: Path, workspace: Path) -> Path:
    root = workspace.resolve()
    destination = path.resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        raise AnalysisError(
            "analysis output must remain inside the workspace",
            code="ANALYSIS_OUTPUT_PATH_ERROR",
        ) from None
    return destination


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


def write_analysis_report(
    report: AnalysisReport,
    path: Path,
    *,
    workspace: Path,
) -> Path:
    """Write an immutable analysis report with an adjacent checksum."""
    destination = _workspace_path(path, workspace)
    checksum = destination.with_name(f"{destination.name}.sha256")
    if destination.exists() or checksum.exists():
        raise AnalysisError(
            "analysis output already exists",
            code="ANALYSIS_OUTPUT_EXISTS",
        )
    payload = canonical_json_bytes(report) + b"\n"
    _replace(destination, payload)
    _replace(checksum, f"{sha256_digest(payload)}\n".encode())
    return destination


__all__ = [
    "FailureAnalysisInput",
    "load_failure_analysis_input",
    "write_analysis_report",
]
