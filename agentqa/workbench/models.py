"""Persistent local-workbench domain models."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from agentqa.models.base import NonEmptyString, ResourceName, StrictModel, UtcDateTime
from agentqa.models.results import ErrorInfo

WorkflowStatus = Literal[
    "DRAFT",
    "READY",
    "RUNNING",
    "WAITING_REVIEW",
    "WAITING_EXTERNAL_REPAIR",
    "INTERRUPTED",
    "FAILED_SAFE",
    "COMPLETED",
    "CANCELLED",
]
WorkflowStepStatus = Literal[
    "QUEUED",
    "RUNNING",
    "WAITING_REVIEW",
    "WAITING_EXTERNAL_REPAIR",
    "INTERRUPTED",
    "FAILED_SAFE",
    "COMPLETED",
    "CANCELLED",
]
WorkflowStepKind = Literal[
    "PREFLIGHT",
    "GENERATE_PLAN",
    "GENERATE_TESTS",
    "REVIEW_TESTS",
    "RUN_CAMPAIGN",
    "RUN_MUTATIONS",
    "RECORD_REPLAY",
    "ANALYZE_FAILURE",
    "COMPARE_RUNS",
    "EVALUATE_GATE",
    "EXPORT_REPORT",
]


class WorkbenchProject(StrictModel):
    """One workspace-confined project known to the local workbench."""

    project_id: ResourceName
    name: NonEmptyString = Field(max_length=200)
    description: str = Field(default="", max_length=4_000)
    workspace_path: str = Field(default=".", max_length=1_000)
    default_spec_refs: dict[str, str] = Field(default_factory=dict, max_length=20)
    created_at: UtcDateTime
    updated_at: UtcDateTime

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("workspace_path must be a safe relative POSIX path")
        return value

    @field_validator("default_spec_refs")
    @classmethod
    def validate_default_spec_refs(cls, value: dict[str, str]) -> dict[str, str]:
        for name, raw_path in value.items():
            path = PurePosixPath(raw_path)
            if not name or path.is_absolute() or ".." in path.parts or "\\" in raw_path:
                raise ValueError("default spec references must use safe relative POSIX paths")
        return value


class ArtifactRef(StrictModel):
    """Small verified reference to an authoritative artifact."""

    artifact_id: NonEmptyString = Field(max_length=200)
    artifact_type: NonEmptyString = Field(max_length=100)
    uri: NonEmptyString = Field(max_length=2_000)
    schema_version: NonEmptyString = Field(max_length=200)
    checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source: NonEmptyString = Field(default="local", max_length=200)
    verification_status: Literal["VERIFIED", "UNVERIFIED", "INVALID"] = "VERIFIED"


class WorkflowStep(StrictModel):
    """One immutable-input attempt inside a workflow."""

    step_id: NonEmptyString
    workflow_id: NonEmptyString
    kind: WorkflowStepKind
    status: WorkflowStepStatus
    attempt: int = Field(ge=1)
    idempotency_key: NonEmptyString = Field(max_length=200)
    input_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_data: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list, max_length=100)
    created_at: UtcDateTime
    started_at: UtcDateTime | None = None
    finished_at: UtcDateTime | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "WorkflowStep":
        if self.status == "RUNNING" and self.started_at is None:
            raise ValueError("running workflow step requires started_at")
        if (
            self.status in {"COMPLETED", "FAILED_SAFE", "CANCELLED", "INTERRUPTED"}
            and self.finished_at is None
        ):
            raise ValueError("terminal workflow step requires finished_at")
        if self.status == "FAILED_SAFE" and self.error is None:
            raise ValueError("failed workflow step requires error")
        return self


class WorkflowRun(StrictModel):
    """Persisted state and immutable identity for a workbench workflow."""

    workflow_id: NonEmptyString
    project_id: ResourceName
    name: NonEmptyString = Field(max_length=200)
    description: str = Field(default="", max_length=4_000)
    status: WorkflowStatus
    revision: int = Field(ge=1)
    idempotency_key: NonEmptyString = Field(max_length=200)
    input_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_snapshot: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: UtcDateTime
    updated_at: UtcDateTime
    started_at: UtcDateTime | None = None
    finished_at: UtcDateTime | None = None
    error: ErrorInfo | None = None
    steps: list[WorkflowStep] = Field(default_factory=list, max_length=1_000)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "WorkflowRun":
        if self.status == "RUNNING" and self.started_at is None:
            raise ValueError("running workflow requires started_at")
        if self.status in {"COMPLETED", "FAILED_SAFE", "CANCELLED"} and self.finished_at is None:
            raise ValueError("terminal workflow requires finished_at")
        if self.status == "FAILED_SAFE" and self.error is None:
            raise ValueError("failed workflow requires error")
        return self


class WorkflowEvent(StrictModel):
    """Monotonically ordered audit event for one workflow."""

    event_id: NonEmptyString
    workflow_id: NonEmptyString
    revision: int = Field(ge=1)
    event_type: NonEmptyString = Field(max_length=100)
    occurred_at: UtcDateTime
    message: str = Field(default="", max_length=2_000)
    step_id: NonEmptyString | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)


class ApprovalDecision(StrictModel):
    """Human decision bound to an immutable subject checksum."""

    approval_id: NonEmptyString
    workflow_id: NonEmptyString
    subject_kind: NonEmptyString = Field(max_length=100)
    subject_id: NonEmptyString = Field(max_length=200)
    subject_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    decision: Literal["APPROVED", "REJECTED", "REQUEST_EDIT"]
    reviewer: NonEmptyString = Field(max_length=200)
    comment: str = Field(default="", max_length=10_000)
    created_at: UtcDateTime


__all__ = [
    "ApprovalDecision",
    "ArtifactRef",
    "WorkbenchProject",
    "WorkflowEvent",
    "WorkflowRun",
    "WorkflowStatus",
    "WorkflowStep",
    "WorkflowStepKind",
    "WorkflowStepStatus",
]
