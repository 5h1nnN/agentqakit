"""Strict HTTP request and asynchronous job models."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from agentqa.models.base import (
    NonEmptyString,
    PositiveInt,
    ResourceName,
    StrictModel,
    UtcDateTime,
    to_camel,
)
from agentqa.models.generation import GenerationGoal
from agentqa.models.results import ErrorInfo
from agentqa.models.specs import AgentTestSpec

ArtifactDiscoverySource = Literal["artifacts", "mcp", "benchmark_results", "reference_results"]


def _default_artifact_sources() -> list[ArtifactDiscoverySource]:
    return ["artifacts", "mcp", "benchmark_results", "reference_results"]


class ApiModel(StrictModel):
    """HTTP model accepting and returning lower-camel aliases."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        populate_by_name=True,
        strict=True,
    )


class RunCreateRequest(ApiModel):
    """Workspace-relative file request for one local background run."""

    project_id: ResourceName = "default"
    tests_path: NonEmptyString
    target_path: NonEmptyString | None = None
    mutations_path: NonEmptyString | None = None
    manifest_path: NonEmptyString | None = None
    random_seed: int = 0
    concurrency: PositiveInt = 1
    max_total_tokens: PositiveInt | None = None
    max_total_cost_usd: float | None = Field(default=None, gt=0)
    capture_trace_content: bool = False
    generate_failure_bundles: bool = True
    judge_enabled: bool = False


class RunJob(ApiModel):
    """Observable lifecycle for a background API execution."""

    run_id: NonEmptyString
    project_id: ResourceName
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
    submitted_at: UtcDateTime
    finished_at: UtcDateTime | None = None
    error: ErrorInfo | None = None


class ComparisonRequest(ApiModel):
    """Stored run identifiers to compare."""

    baseline_run_id: NonEmptyString
    candidate_run_id: NonEmptyString


class GateEvaluateRequest(ApiModel):
    """Stored candidate, optional baseline, and workspace-relative gate spec."""

    gate_path: NonEmptyString
    candidate_run_id: NonEmptyString
    baseline_run_id: NonEmptyString | None = None


class ReviewUpdateRequest(ApiModel):
    """Mutable human judgement over one immutable finding."""

    status: Literal["unreviewed", "confirmed", "false_positive", "resolved"]
    note: str = Field(default="", max_length=10_000)


class ProjectCreateRequest(ApiModel):
    """Create one workspace-confined workbench project."""

    project_id: ResourceName
    name: NonEmptyString = Field(max_length=200)
    description: str = Field(default="", max_length=4_000)
    workspace_path: str = Field(default=".", max_length=1_000)
    default_spec_refs: dict[str, str] = Field(default_factory=dict, max_length=20)
    tags: list[ResourceName] = Field(default_factory=list, max_length=20)

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("workspacePath must be a safe relative POSIX path")
        return value

    @field_validator("default_spec_refs")
    @classmethod
    def validate_default_spec_refs(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not name
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or "\\" in path
            for name, path in value.items()
        ):
            raise ValueError("defaultSpecRefs must use safe relative POSIX paths")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("tags must not contain duplicates")
        return value


class WorkflowCreateRequest(ApiModel):
    """Create an idempotent persistent workflow container."""

    project_id: ResourceName
    name: NonEmptyString = Field(max_length=200)
    description: str = Field(default="", max_length=4_000)
    idempotency_key: NonEmptyString = Field(max_length=200)
    draft: bool = False


class ArtifactImportRequest(ApiModel):
    """Explicit allowlisted artifact discovery request."""

    sources: list[ArtifactDiscoverySource] = Field(
        default_factory=_default_artifact_sources,
        min_length=1,
        max_length=4,
    )

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("artifact sources must not contain duplicates")
        return value


class CodeEvidenceRequest(ApiModel):
    """Explicit bounded file and line selection."""

    path: NonEmptyString = Field(max_length=2_000)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    reason: NonEmptyString = Field(max_length=4_000)


class GenerationSessionCreateRequest(ApiModel):
    """Create and immediately queue the plan phase of a generation session."""

    idempotency_key: NonEmptyString = Field(max_length=200)
    goal: GenerationGoal | None = None
    goal_path: NonEmptyString | None = Field(default=None, max_length=2_000)
    manifest_path: NonEmptyString | None = Field(default=None, max_length=2_000)
    mutations_path: NonEmptyString | None = Field(default=None, max_length=2_000)
    evidence_mode: Literal["MANIFEST_ONLY", "MANIFEST_AND_CODE"] = "MANIFEST_ONLY"
    code_evidence_hashes: list[str] = Field(default_factory=list, max_length=10)
    max_output_tokens: int = Field(default=4_096, ge=64, le=16_384)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)


class GenerateCandidatesRequest(ApiModel):
    idempotency_key: NonEmptyString = Field(max_length=200)
    plan_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class CandidateReviewRequest(ApiModel):
    subject_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    decision: Literal["APPROVED", "REJECTED", "REQUEST_EDIT"]
    reviewer: NonEmptyString = Field(max_length=200)
    comment: str = Field(default="", max_length=10_000)


class BatchCandidateReviewRequest(ApiModel):
    case_ids: list[ResourceName] = Field(min_length=1, max_length=50)
    reviewer: NonEmptyString = Field(max_length=200)
    comment: str = Field(default="", max_length=10_000)


class CandidateRevisionRequest(ApiModel):
    spec: AgentTestSpec
    editor: NonEmptyString = Field(max_length=200)
    diff_summary: NonEmptyString = Field(max_length=4_000)


class GeneratedCampaignTestRequest(ApiModel):
    session_id: NonEmptyString
    case_id: ResourceName
    spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class CampaignTargetRequest(ApiModel):
    role: Literal["target", "baseline", "candidate", "defect", "fixed"]
    path: NonEmptyString = Field(max_length=2_000)


class CampaignCreateRequest(ApiModel):
    """User choices that the server resolves into an immutable CampaignConfig."""

    idempotency_key: NonEmptyString = Field(max_length=200)
    golden_tests_path: NonEmptyString | None = Field(default=None, max_length=2_000)
    golden_case_ids: list[ResourceName] = Field(default_factory=list, max_length=1_000)
    generated_tests: list[GeneratedCampaignTestRequest] = Field(
        default_factory=list, max_length=1_000
    )
    targets: list[CampaignTargetRequest] = Field(min_length=1, max_length=20)
    manifest_path: NonEmptyString | None = Field(default=None, max_length=2_000)
    mutations_path: NonEmptyString | None = Field(default=None, max_length=2_000)
    replay_mode: Literal["off", "record", "replay", "hybrid"] = "off"
    replay_cassette: str | None = Field(default=None, max_length=2_000)
    judge_enabled: bool = False
    trials: int | None = Field(default=None, ge=1, le=100)
    random_seed: int = 0
    concurrency: int = Field(default=1, ge=1, le=32)
    item_timeout_seconds: float = Field(default=120.0, gt=0, le=3_600)
    evaluator_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_total_cost_usd: float | None = Field(default=None, gt=0)
    max_duration_seconds: float | None = Field(default=None, gt=0, le=86_400)
    allow_side_effects: bool = False


class CampaignConfirmRequest(ApiModel):
    challenge_id: NonEmptyString
    phrase: NonEmptyString = Field(max_length=100)
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    reviewer: NonEmptyString = Field(max_length=200)
    comment: str = Field(default="", max_length=10_000)


class CampaignStartRequest(ApiModel):
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    idempotency_key: NonEmptyString = Field(max_length=200)
