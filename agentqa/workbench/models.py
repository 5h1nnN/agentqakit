"""Persistent local-workbench domain models."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from agentqa.llm import LLMCallEvidence
from agentqa.models.base import NonEmptyString, ResourceName, StrictModel, UtcDateTime
from agentqa.models.generation import CodeEvidence, GenerationGoalSummary, GenerationRisk
from agentqa.models.manifest import RunManifestSnapshot
from agentqa.models.results import ErrorInfo, RunBudgetSummary, RunSummary
from agentqa.models.specs import AgentTestSpec

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
    tags: list[ResourceName] = Field(default_factory=list, max_length=20)
    recent_workflow_id: NonEmptyString | None = None
    last_manifest_snapshot: RunManifestSnapshot | None = None
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

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("project tags must be unique")
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


PreflightStatus = Literal["PASS", "REVIEW", "BLOCK"]


class PreflightCheck(StrictModel):
    """One bounded project readiness observation."""

    check_id: ResourceName
    category: Literal[
        "schema",
        "path",
        "dependency",
        "entrypoint",
        "provider",
        "tool_risk",
        "side_effect",
        "source",
    ]
    status: PreflightStatus
    message: NonEmptyString = Field(max_length=1_000)
    path: str | None = Field(default=None, max_length=1_000)


class PreflightReport(StrictModel):
    """Read-only readiness result with locally observed source identity."""

    project_id: ResourceName
    checked_at: UtcDateTime
    executable: bool
    requires_approval: bool
    checks: list[PreflightCheck] = Field(max_length=200)
    manifest_snapshot: RunManifestSnapshot | None = None


class CatalogArtifact(StrictModel):
    """Verified, immutable index entry pointing at an existing artifact file."""

    artifact_id: NonEmptyString = Field(max_length=200)
    project_id: ResourceName
    artifact_type: NonEmptyString = Field(max_length=100)
    source: Literal["CLI", "API", "MCP", "WORKFLOW", "BENCHMARK", "REFERENCE", "IMPORTED"]
    path: NonEmptyString = Field(max_length=2_000)
    schema_version: NonEmptyString = Field(max_length=200)
    checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    created_at: UtcDateTime
    workflow_id: NonEmptyString | None = None
    run_id: NonEmptyString | None = None
    availability: Literal["AVAILABLE", "MISSING", "INVALID"] = "AVAILABLE"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ArtifactImportIssue(StrictModel):
    path: str = Field(max_length=2_000)
    code: NonEmptyString = Field(max_length=100)
    message: NonEmptyString = Field(max_length=1_000)


class ArtifactImportResult(StrictModel):
    project_id: ResourceName
    scanned: int = Field(ge=0)
    imported: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    rejected: int = Field(ge=0)
    runs_registered: int = Field(ge=0)
    issues: list[ArtifactImportIssue] = Field(default_factory=list, max_length=100)


class WorkspaceEntry(StrictModel):
    path: NonEmptyString = Field(max_length=2_000)
    name: NonEmptyString = Field(max_length=500)
    kind: Literal["file", "directory"]
    size_bytes: int | None = Field(default=None, ge=0)


class EvidenceBoundaryPreview(StrictModel):
    project_id: ResourceName
    manifest_sections: list[NonEmptyString] = Field(default_factory=list, max_length=50)
    code_evidence: list[CodeEvidence] = Field(default_factory=list, max_length=10)
    evidence_types: list[NonEmptyString] = Field(default_factory=list, max_length=50)
    excluded: list[NonEmptyString] = Field(default_factory=list, max_length=50)
    provider: NonEmptyString | None = None
    model: NonEmptyString | None = None
    max_input_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    evidence_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


GenerationSessionStatus = Literal[
    "PLANNING",
    "PLAN_READY",
    "GENERATING",
    "CANDIDATES_READY",
    "FAILED_SAFE",
]


class GenerationSession(StrictModel):
    """Persisted two-phase generation state without raw prompts or model output."""

    session_id: NonEmptyString
    workflow_id: NonEmptyString
    project_id: ResourceName
    status: GenerationSessionStatus
    idempotency_key: NonEmptyString = Field(max_length=200)
    input_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    goal: GenerationGoalSummary
    manifest_path: NonEmptyString = Field(max_length=2_000)
    mutations_path: str | None = Field(default=None, max_length=2_000)
    evidence_mode: Literal["MANIFEST_ONLY", "MANIFEST_AND_CODE"]
    code_evidence_hashes: list[str] = Field(default_factory=list, max_length=10)
    max_output_tokens: int = Field(ge=64, le=16_384)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    consumed_tokens: int = Field(default=0, ge=0)
    consumed_cost_usd: float | None = Field(default=None, ge=0)
    plan_path: str | None = Field(default=None, max_length=2_000)
    plan_checksum: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    report_path: str | None = Field(default=None, max_length=2_000)
    report_checksum: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    llm_calls: list[LLMCallEvidence] = Field(default_factory=list, max_length=2)
    error: ErrorInfo | None = None
    created_at: UtcDateTime
    updated_at: UtcDateTime

    @model_validator(mode="after")
    def validate_generation_state(self) -> "GenerationSession":
        if self.evidence_mode == "MANIFEST_ONLY" and self.code_evidence_hashes:
            raise ValueError("manifest-only generation cannot include code evidence")
        if self.status in {"PLAN_READY", "GENERATING", "CANDIDATES_READY"} and (
            self.plan_path is None or self.plan_checksum is None
        ):
            raise ValueError("generation session state requires a verified plan")
        if self.status == "CANDIDATES_READY" and (
            self.report_path is None or self.report_checksum is None
        ):
            raise ValueError("candidate-ready generation requires a verified report")
        if self.status == "FAILED_SAFE" and self.error is None:
            raise ValueError("failed generation session requires an error")
        return self


class GeneratedTestRevision(StrictModel):
    """Immutable human edit layered over an immutable generated test."""

    revision_id: NonEmptyString
    session_id: NonEmptyString
    case_id: ResourceName
    revision: int = Field(ge=1)
    parent_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    spec: AgentTestSpec
    editor: NonEmptyString = Field(max_length=200)
    diff_summary: NonEmptyString = Field(max_length=4_000)
    created_at: UtcDateTime

    @model_validator(mode="after")
    def validate_revision_identity(self) -> "GeneratedTestRevision":
        if self.spec.metadata.name != self.case_id:
            raise ValueError("revision case id must match AgentTest metadata.name")
        return self


class CandidateDiagnostic(StrictModel):
    code: NonEmptyString = Field(max_length=100)
    status: Literal["PASS", "WARN", "BLOCK"]
    message: NonEmptyString = Field(max_length=1_000)


class SimilarityHint(StrictModel):
    source: NonEmptyString = Field(max_length=2_000)
    score: float = Field(ge=0, le=1)


class GeneratedCandidateView(StrictModel):
    """Review projection for one original candidate and its latest optional revision."""

    case_id: ResourceName
    risk: GenerationRisk
    review_status: NonEmptyString
    rationale: NonEmptyString
    assertion_basis: list[NonEmptyString]
    coverage_item_ids: list[ResourceName]
    source_refs: list[NonEmptyString]
    original_spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    current_spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    spec: AgentTestSpec
    latest_revision: GeneratedTestRevision | None = None
    decision: ApprovalDecision | None = None
    diagnostics: list[CandidateDiagnostic]
    similarity_hints: list[SimilarityHint] = Field(default_factory=list, max_length=10)
    campaign_eligible: bool


CampaignStatus = Literal[
    "DRAFT",
    "AWAITING_CONFIRMATION",
    "QUEUED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "INCOMPLETE",
    "INTERRUPTED",
    "CANCELLED",
]
CampaignSubrunStatus = Literal[
    "QUEUED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "INCOMPLETE",
    "INTERRUPTED",
    "CANCELLED",
]


class CampaignTestSelection(StrictModel):
    """Immutable identity for one selected test revision."""

    source: Literal["golden", "generated"]
    case_id: ResourceName
    spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_ref: NonEmptyString = Field(max_length=2_000)
    risk: Literal["low", "medium", "high", "critical"] = "low"


class CampaignTarget(StrictModel):
    """One independently executed target in a campaign matrix."""

    role: Literal["target", "baseline", "candidate", "defect", "fixed"]
    path: NonEmptyString = Field(max_length=2_000)
    name: ResourceName
    spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class CampaignConfig(StrictModel):
    """Frozen, fully resolved campaign configuration bound to approvals."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    tests: list[CampaignTestSelection] = Field(min_length=1, max_length=1_000)
    targets: list[CampaignTarget] = Field(min_length=1, max_length=20)
    manifest_path: NonEmptyString
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_snapshot: RunManifestSnapshot
    mutations_path: NonEmptyString | None = None
    mutation_ids: list[ResourceName] = Field(default_factory=list, max_length=500)
    replay_mode: Literal["off", "record", "replay", "hybrid"] = "off"
    replay_cassette: str | None = Field(default=None, max_length=2_000)
    judge_enabled: bool = False
    judge_provider: NonEmptyString | None = None
    trials: int | None = Field(default=None, ge=1, le=100)
    random_seed: int = 0
    concurrency: int = Field(default=1, ge=1, le=32)
    item_timeout_seconds: float = Field(default=120.0, gt=0, le=3_600)
    evaluator_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_total_cost_usd: float | None = Field(default=None, gt=0)
    max_duration_seconds: float | None = Field(default=None, gt=0, le=86_400)
    allow_side_effects: bool = False

    @model_validator(mode="after")
    def validate_replay(self) -> "CampaignConfig":
        if self.replay_mode in {"replay", "hybrid"} and self.replay_cassette is None:
            raise ValueError("replay and hybrid modes require a cassette")
        if self.replay_mode not in {"replay", "hybrid"} and self.replay_cassette is not None:
            raise ValueError("a replay cassette is only valid in replay or hybrid mode")
        if self.judge_enabled and self.judge_provider is None:
            raise ValueError("enabled judge requires an available provider")
        return self


class CampaignRiskSummary(StrictModel):
    """Bounded confirmation information calculated by the server."""

    case_count: int = Field(ge=1)
    target_count: int = Field(ge=1)
    mutation_count: int = Field(ge=0)
    max_trials: int = Field(ge=1)
    high_risk_cases: list[ResourceName] = Field(default_factory=list)
    side_effect_tools: list[ResourceName] = Field(default_factory=list)
    external_side_effects: list[ResourceName] = Field(default_factory=list)
    irreversible_side_effects: list[ResourceName] = Field(default_factory=list)
    requires_confirmation: bool
    reasons: list[NonEmptyString] = Field(default_factory=list)


class CampaignSubrun(StrictModel):
    """Persistent execution checkpoint for one target in a campaign."""

    subrun_id: NonEmptyString
    campaign_id: NonEmptyString
    target: CampaignTarget
    run_id: NonEmptyString
    prior_run_ids: list[NonEmptyString] = Field(default_factory=list)
    attempt: int = Field(default=1, ge=1)
    status: CampaignSubrunStatus
    planned_trials: int = Field(ge=1)
    completed_trials: int = Field(default=0, ge=0)
    current_case_id: str | None = None
    replay_hits: int = Field(default=0, ge=0)
    replay_misses: int = Field(default=0, ge=0)
    live_fallbacks: int = Field(default=0, ge=0)
    judge_evaluations: int = Field(default=0, ge=0)
    mutation_applications: int = Field(default=0, ge=0)
    mutation_killed: int = Field(default=0, ge=0)
    mutation_survived: int = Field(default=0, ge=0)
    mutation_not_reached: int = Field(default=0, ge=0)
    mutation_invalid: int = Field(default=0, ge=0)
    summary: RunSummary | None = None
    budget: RunBudgetSummary | None = None
    artifact_path: str | None = None
    started_at: UtcDateTime | None = None
    finished_at: UtcDateTime | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> "CampaignSubrun":
        if self.completed_trials > self.planned_trials:
            raise ValueError("completed trials cannot exceed planned trials")
        if self.status == "RUNNING" and self.started_at is None:
            raise ValueError("running subrun requires started_at")
        if (
            self.status in {"COMPLETED", "FAILED", "INCOMPLETE", "INTERRUPTED", "CANCELLED"}
            and self.finished_at is None
        ):
            raise ValueError("terminal subrun requires finished_at")
        return self


class Campaign(StrictModel):
    """Durable campaign aggregate with immutable configuration and live progress."""

    campaign_id: NonEmptyString
    workflow_id: NonEmptyString
    project_id: ResourceName
    idempotency_key: NonEmptyString = Field(max_length=200)
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config: CampaignConfig
    risk: CampaignRiskSummary
    status: CampaignStatus
    approval_id: str | None = None
    workflow_step_id: str | None = None
    completed_trials: int = Field(default=0, ge=0)
    total_trials: int = Field(ge=1)
    created_at: UtcDateTime
    updated_at: UtcDateTime
    started_at: UtcDateTime | None = None
    finished_at: UtcDateTime | None = None
    error: ErrorInfo | None = None
    subruns: list[CampaignSubrun] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_campaign(self) -> "Campaign":
        if self.completed_trials > self.total_trials:
            raise ValueError("campaign completed trials cannot exceed total trials")
        if self.status == "RUNNING" and self.started_at is None:
            raise ValueError("running campaign requires started_at")
        if (
            self.status
            in {
                "COMPLETED",
                "FAILED",
                "INCOMPLETE",
                "INTERRUPTED",
                "CANCELLED",
            }
            and self.finished_at is None
        ):
            raise ValueError("terminal campaign requires finished_at")
        return self


class CampaignConfirmationChallenge(StrictModel):
    """Single-use confirmation challenge bound to one exact config hash."""

    challenge_id: NonEmptyString
    campaign_id: NonEmptyString
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    phrase_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    status: Literal["PENDING", "CONSUMED", "EXPIRED"]
    created_at: UtcDateTime
    expires_at: UtcDateTime
    consumed_at: UtcDateTime | None = None


__all__ = [
    "ApprovalDecision",
    "ArtifactImportIssue",
    "ArtifactImportResult",
    "ArtifactRef",
    "Campaign",
    "CampaignConfig",
    "CampaignConfirmationChallenge",
    "CampaignRiskSummary",
    "CampaignStatus",
    "CampaignSubrun",
    "CampaignSubrunStatus",
    "CampaignTarget",
    "CampaignTestSelection",
    "CandidateDiagnostic",
    "CatalogArtifact",
    "EvidenceBoundaryPreview",
    "GeneratedCandidateView",
    "GeneratedTestRevision",
    "GenerationSession",
    "GenerationSessionStatus",
    "PreflightCheck",
    "PreflightReport",
    "PreflightStatus",
    "SimilarityHint",
    "WorkbenchProject",
    "WorkflowEvent",
    "WorkflowRun",
    "WorkflowStatus",
    "WorkflowStep",
    "WorkflowStepKind",
    "WorkflowStepStatus",
    "WorkspaceEntry",
]
