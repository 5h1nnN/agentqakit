"""Public models for manifest-driven AI test generation."""

import hashlib
import json
from pathlib import PurePosixPath
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, StringConstraints, model_validator

from agentqa.llm.provider import LLMCallEvidence
from agentqa.models.base import (
    NonEmptyString,
    PositiveInt,
    ResourceName,
    SpecModel,
    StrictModel,
    UtcDateTime,
)
from agentqa.models.specs import AgentTestSpec, ApiVersion, Metadata

GENERATION_PLAN_SCHEMA_VERSION: Literal["agentqa.generation-plan/v1alpha1"] = (
    "agentqa.generation-plan/v1alpha1"
)
GENERATION_REPORT_SCHEMA_VERSION: Literal["agentqa.generation-report/v1alpha1"] = (
    "agentqa.generation-report/v1alpha1"
)

ScenarioCategory: TypeAlias = Literal[
    "normal",
    "boundary",
    "fault",
    "security",
    "efficiency",
    "hitl",
]
GenerationRisk: TypeAlias = Literal["low", "medium", "high", "critical"]
ProvenanceLevel: TypeAlias = Literal[
    "declared",
    "discovered",
    "inferred",
    "code",
]
ReviewStatus: TypeAlias = Literal["READY", "REVIEW_REQUIRED", "APPROVED"]
Sha256Digest: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
BoundedText: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
]
EvidencePath: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1_024),
]
CodeContent: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=1, max_length=16_384),
]


def _default_categories() -> list[ScenarioCategory]:
    return ["normal", "boundary", "fault", "security", "efficiency", "hitl"]


def _default_risk_levels() -> list[GenerationRisk]:
    return ["low", "medium", "high"]


class CodeEvidence(SpecModel):
    """A caller-selected, bounded code excerpt; never a filesystem read request."""

    path: EvidencePath
    start_line: PositiveInt
    end_line: PositiveInt
    content_hash: Sha256Digest
    reason: BoundedText
    content: CodeContent

    @model_validator(mode="after")
    def validate_reference(self) -> "CodeEvidence":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError("code evidence path must be a safe relative POSIX path")
        if self.end_line < self.start_line:
            raise ValueError("code evidence endLine must not be before startLine")
        content_hash = f"sha256:{hashlib.sha256(self.content.encode('utf-8')).hexdigest()}"
        if self.content_hash != content_hash:
            raise ValueError("code evidence contentHash does not match content")
        return self


class GenerationGoal(SpecModel):
    """User-owned generation scope, risk envelope, and case budget."""

    api_version: ApiVersion = Field(default="agentqa.dev/v1alpha1", alias="apiVersion")
    kind: Literal["GenerationGoal"] = "GenerationGoal"
    metadata: Metadata
    categories: list[ScenarioCategory] = Field(
        default_factory=_default_categories,
        min_length=1,
        max_length=6,
    )
    risk_levels: list[GenerationRisk] = Field(
        default_factory=_default_risk_levels,
        min_length=1,
        max_length=4,
    )
    target_capabilities: list[ResourceName] = Field(default_factory=list, max_length=200)
    target_policies: list[ResourceName] = Field(default_factory=list, max_length=200)
    allowed_mutations: list[ResourceName] = Field(default_factory=list, max_length=200)
    max_cases: Annotated[int, Field(ge=1, le=50)] = 10
    instructions: BoundedText | None = None
    code_evidence: list[CodeEvidence] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_goal(self) -> "GenerationGoal":
        for label, values in (
            ("categories", self.categories),
            ("riskLevels", self.risk_levels),
            ("targetCapabilities", self.target_capabilities),
            ("targetPolicies", self.target_policies),
            ("allowedMutations", self.allowed_mutations),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"generation goal {label} must not contain duplicates")
        evidence_hashes = [item.content_hash for item in self.code_evidence]
        if len(evidence_hashes) != len(set(evidence_hashes)):
            raise ValueError("generation goal codeEvidence hashes must be unique")
        encoded = json.dumps(
            self.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 200 * 1024:
            raise ValueError("GenerationGoal exceeds 204800 bytes")
        return self


class GenerationGoalSummary(StrictModel):
    """Persisted goal fields without raw code contents."""

    categories: list[ScenarioCategory] = Field(min_length=1, max_length=6)
    risk_levels: list[GenerationRisk] = Field(min_length=1, max_length=4)
    target_capabilities: list[ResourceName] = Field(default_factory=list, max_length=200)
    target_policies: list[ResourceName] = Field(default_factory=list, max_length=200)
    allowed_mutations: list[ResourceName] = Field(default_factory=list, max_length=200)
    max_cases: Annotated[int, Field(ge=1, le=50)]
    instructions: BoundedText | None = None
    code_evidence_hashes: list[Sha256Digest] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_summary(self) -> "GenerationGoalSummary":
        for label, values in (
            ("categories", self.categories),
            ("risk_levels", self.risk_levels),
            ("target_capabilities", self.target_capabilities),
            ("target_policies", self.target_policies),
            ("allowed_mutations", self.allowed_mutations),
            ("code_evidence_hashes", self.code_evidence_hashes),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"generation goal summary {label} must not contain duplicates")
        return self


class PlanItem(StrictModel):
    """One capability-risk-scenario cell proposed before test generation."""

    id: ResourceName
    title: BoundedText
    category: ScenarioCategory
    risk: GenerationRisk
    capabilities: list[ResourceName] = Field(default_factory=list, max_length=50)
    policies: list[ResourceName] = Field(default_factory=list, max_length=50)
    tools: list[ResourceName] = Field(default_factory=list, max_length=50)
    mutation: ResourceName | None = None
    rationale: BoundedText
    source_refs: list[NonEmptyString] = Field(min_length=1, max_length=50)
    provenance: list[ProvenanceLevel] = Field(min_length=1, max_length=4)
    requires_human: bool = False

    @model_validator(mode="after")
    def validate_lists(self) -> "PlanItem":
        for label, values in (
            ("capabilities", self.capabilities),
            ("policies", self.policies),
            ("tools", self.tools),
            ("source_refs", self.source_refs),
            ("provenance", self.provenance),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"plan item {label} must not contain duplicates")
        return self


class GenerationPlan(StrictModel):
    """Validated first-phase output and its reproducibility evidence."""

    schema_version: Literal["agentqa.generation-plan/v1alpha1"] = GENERATION_PLAN_SCHEMA_VERSION
    generation_id: NonEmptyString
    created_at: UtcDateTime
    manifest_hash: Sha256Digest
    goal_hash: Sha256Digest
    goal: GenerationGoalSummary
    items: list[PlanItem] = Field(min_length=1, max_length=50)
    questions: list[BoundedText] = Field(default_factory=list, max_length=50)
    warnings: list[BoundedText] = Field(default_factory=list, max_length=50)
    llm_call: LLMCallEvidence

    @model_validator(mode="after")
    def validate_plan(self) -> "GenerationPlan":
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("generation plan item ids must be unique")
        if len(self.items) > self.goal.max_cases:
            raise ValueError("generation plan exceeds the goal maxCases")
        return self


class GeneratedCaseDraft(StrictModel):
    """Second-phase LLM output before policy-controlled labels are applied."""

    spec: AgentTestSpec
    rationale: BoundedText
    assertion_basis: list[BoundedText] = Field(min_length=1, max_length=50)
    coverage_item_ids: list[ResourceName] = Field(min_length=1, max_length=50)
    source_refs: list[NonEmptyString] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_lists(self) -> "GeneratedCaseDraft":
        if len(self.coverage_item_ids) != len(set(self.coverage_item_ids)):
            raise ValueError("generated case coverageItemIds must be unique")
        if len(self.source_refs) != len(set(self.source_refs)):
            raise ValueError("generated case sourceRefs must be unique")
        return self


class GeneratedCase(StrictModel):
    """A validated draft test plus its explanation and review state."""

    case_id: ResourceName
    spec: AgentTestSpec
    rationale: BoundedText
    assertion_basis: list[BoundedText] = Field(min_length=1, max_length=50)
    coverage_item_ids: list[ResourceName] = Field(min_length=1, max_length=50)
    source_refs: list[NonEmptyString] = Field(min_length=1, max_length=50)
    risk: GenerationRisk
    review_status: ReviewStatus

    @model_validator(mode="after")
    def validate_case_identity(self) -> "GeneratedCase":
        if self.case_id != self.spec.metadata.name:
            raise ValueError("generated case id must match AgentTest metadata.name")
        return self


class CoverageRecord(StrictModel):
    """Tests covering one declared capability or policy."""

    subject_type: Literal["capability", "policy"]
    subject: ResourceName
    case_ids: list[ResourceName] = Field(default_factory=list, max_length=50)


class GenerationReport(StrictModel):
    """Final generation coverage, gaps, warnings, and AI evidence."""

    schema_version: Literal["agentqa.generation-report/v1alpha1"] = GENERATION_REPORT_SCHEMA_VERSION
    generation_id: NonEmptyString
    generated_at: UtcDateTime
    manifest_hash: Sha256Digest
    goal_hash: Sha256Digest
    plan_hash: Sha256Digest
    cases: list[GeneratedCase] = Field(min_length=1, max_length=50)
    coverage: list[CoverageRecord] = Field(default_factory=list, max_length=400)
    gaps: list[BoundedText] = Field(default_factory=list, max_length=400)
    warnings: list[BoundedText] = Field(default_factory=list, max_length=100)
    code_evidence_hashes: list[Sha256Digest] = Field(default_factory=list, max_length=10)
    llm_calls: list[LLMCallEvidence] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_report(self) -> "GenerationReport":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("generation report case ids must be unique")
        subjects = [(item.subject_type, item.subject) for item in self.coverage]
        if len(subjects) != len(set(subjects)):
            raise ValueError("generation report coverage subjects must be unique")
        return self


class GenerationPlanDraft(StrictModel):
    """Strict response schema used for first-phase LLM output."""

    items: list[PlanItem] = Field(min_length=1, max_length=50)
    questions: list[BoundedText] = Field(default_factory=list, max_length=50)
    warnings: list[BoundedText] = Field(default_factory=list, max_length=50)


class GeneratedSuiteDraft(StrictModel):
    """Strict response schema used for second-phase LLM output."""

    cases: list[GeneratedCaseDraft] = Field(min_length=1, max_length=50)
