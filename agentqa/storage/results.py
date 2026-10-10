"""Result-store protocol and shared evidence helpers."""

import json
import re
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from agentqa.errors import AgentQAError
from agentqa.models import RunResult, TestResult, TraceRecord
from agentqa.models.storage import FindingReview, StoredFinding, StoredRun, StoredRunSummary
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest

RESULT_STORE_SCHEMA_VERSION = "agentqa.result-store/v1alpha1"
_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^sha256:([0-9a-f]{64})$")


class ResultStoreError(AgentQAError):
    """Expected persistence or evidence-loading failure."""

    def __init__(self, message: str, *, code: str = "RESULT_STORE_ERROR") -> None:
        super().__init__(message, code=code)


class ResultStore(Protocol):
    """Asynchronous persistence boundary shared by API and local consumers."""

    async def initialize(self) -> None: ...

    async def save_run(
        self,
        run: RunResult,
        *,
        project_id: str = "default",
        artifact_path: str | Path | None = None,
    ) -> StoredRun: ...

    async def get_run(self, run_id: str) -> StoredRun | None: ...

    async def list_runs(self, project_id: str = "default") -> list[StoredRunSummary]: ...

    async def list_results(self, run_id: str) -> list[TestResult]: ...

    async def get_trace(self, trace_id: str) -> TraceRecord | None: ...

    async def list_findings(self, run_id: str) -> list[StoredFinding]: ...

    async def get_review(self, finding_id: str) -> FindingReview | None: ...

    async def save_review(self, review: FindingReview) -> None: ...

    async def close(self) -> None: ...


def validate_project_id(project_id: str) -> str:
    """Return a safe project identifier or raise a public store error."""
    if not _RESOURCE_ID.fullmatch(project_id):
        raise ResultStoreError(
            "project_id must be a valid resource name",
            code="RESULT_STORE_PROJECT_ID_ERROR",
        )
    return project_id


def validate_run_id(run_id: str) -> str:
    """Return a path-safe run identifier shared by every store implementation."""
    if not _RESOURCE_ID.fullmatch(run_id):
        raise ResultStoreError(
            "run_id must be a valid resource name",
            code="RESULT_STORE_RUN_ID_ERROR",
        )
    return run_id


def finding_filename(finding_id: str) -> str:
    """Encode a namespaced digest as a portable review filename."""
    match = _SHA256.fullmatch(finding_id)
    if match is None:
        raise ResultStoreError("invalid finding_id", code="RESULT_STORE_FINDING_ID_ERROR")
    return f"sha256-{match.group(1)}.json"


def findings_for(run: RunResult) -> list[StoredFinding]:
    """Derive stable finding IDs from immutable trial violations."""
    findings: list[StoredFinding] = []
    for result in run.results:
        for index, violation in enumerate(result.violations):
            material = canonical_json_bytes(
                {
                    "run_id": run.run_id,
                    "trial_id": result.trial_id,
                    "index": index,
                    "type": violation.type,
                }
            )
            findings.append(
                StoredFinding(
                    finding_id=sha256_digest(material),
                    run_id=run.run_id,
                    test_case_id=result.test_case_id,
                    trial_id=result.trial_id,
                    type=violation.type,
                    message=violation.message,
                    span_id=violation.span_id,
                    details=violation.details,
                )
            )
    return findings


def read_trace_artifact(artifact_root: str | Path, relative_ref: str) -> TraceRecord:
    """Read one trace while enforcing containment, checksum, and schema validity."""
    root = Path(artifact_root).resolve()
    relative = Path(relative_ref)
    if relative.is_absolute():
        raise ResultStoreError("trace reference must be relative", code="TRACE_PATH_ERROR")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ResultStoreError("trace reference escapes its run", code="TRACE_PATH_ERROR") from None
    checksum_path = path.with_name(f"{path.name}.sha256")
    try:
        payload = path.read_bytes()
        expected = checksum_path.read_text(encoding="utf-8").strip()
    except OSError:
        raise ResultStoreError("trace artifact is missing", code="TRACE_NOT_FOUND") from None
    if expected != sha256_digest(payload):
        raise ResultStoreError("trace artifact checksum mismatch", code="TRACE_CHECKSUM_ERROR")
    try:
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise TypeError
        document.pop("schema_version", None)
        return TraceRecord.model_validate_json(json.dumps(document))
    except (json.JSONDecodeError, TypeError, ValidationError):
        raise ResultStoreError(
            "trace artifact does not match the expected schema",
            code="TRACE_FORMAT_ERROR",
        ) from None
