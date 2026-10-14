"""Models for bounded failure evidence and non-authoritative AI analysis."""

from typing import Annotated, Final, Literal, TypeAlias

from pydantic import Field, JsonValue, StringConstraints, model_validator

from agentqa.llm import LLMCallEvidence
from agentqa.models.base import NonEmptyString, ResourceName, StrictModel, UnitInterval, UtcDateTime

ANALYSIS_REPORT_SCHEMA_VERSION: Final = "agentqa.analysis-report/v1alpha1"

AnalysisText: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
]
EvidenceId: TypeAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
Sha256Digest: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
EvidenceKind: TypeAlias = Literal[
    "code",
    "scenario",
    "target",
    "manifest",
    "evaluation",
    "state",
    "tool",
    "error",
    "trace",
]


class EvidenceReference(StrictModel):
    """Stable reference retained in an analysis report without full raw evidence."""

    evidence_id: EvidenceId
    kind: EvidenceKind
    source_ref: NonEmptyString
    content_hash: Sha256Digest
    summary: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
    ]
    truncated: bool = False


class EvidenceRecord(EvidenceReference):
    """One bounded, redacted evidence item supplied to an AI operation."""

    data: JsonValue


class EvidenceBundle(StrictModel):
    """Deterministic evidence package shared by Judge and Failure Analyst."""

    run_id: NonEmptyString
    test_case_id: ResourceName
    trial_id: NonEmptyString
    trace_status: Literal["complete", "partial", "missing"]
    items: list[EvidenceRecord] = Field(min_length=1, max_length=100)
    missing_evidence: list[AnalysisText] = Field(default_factory=list, max_length=100)
    blocking_gaps: list[AnalysisText] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_bundle(self) -> "EvidenceBundle":
        ids = [item.evidence_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence ids must be unique")
        if len(self.missing_evidence) != len(set(self.missing_evidence)):
            raise ValueError("missing evidence entries must be unique")
        if len(self.blocking_gaps) != len(set(self.blocking_gaps)):
            raise ValueError("blocking evidence gaps must be unique")
        return self


class ObservedFact(StrictModel):
    """A direct observation with explicit evidence references."""

    statement: AnalysisText
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=20)


class TrajectoryDivergence(StrictModel):
    """Earliest observed anomaly or departure from an expected trajectory."""

    description: AnalysisText
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=20)
    span_id: NonEmptyString | None = None


class RootCauseCandidate(StrictModel):
    """Non-authoritative root cause hypothesis grounded in evidence."""

    id: ResourceName
    hypothesis: AnalysisText
    confidence: UnitInterval
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=20)
    counter_evidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=20)
    reasoning: AnalysisText


class RegressionTestSuggestion(StrictModel):
    """Suggested future test, not an automatically executable specification."""

    title: AnalysisText
    rationale: AnalysisText
    expected_behavior: AnalysisText
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=20)


class AnalysisDraft(StrictModel):
    """Strict LLM output before immutable trial identity and evidence are attached."""

    observed_facts: list[ObservedFact] = Field(min_length=1, max_length=50)
    earliest_divergence: TrajectoryDivergence | None = None
    root_causes: list[RootCauseCandidate] = Field(min_length=1, max_length=20)
    additional_evidence_needed: list[AnalysisText] = Field(default_factory=list, max_length=50)
    regression_tests: list[RegressionTestSuggestion] = Field(default_factory=list, max_length=20)
    repair_directions: list[AnalysisText] = Field(default_factory=list, max_length=20)


class AnalysisReport(StrictModel):
    """Separate, non-authoritative explanation of one immutable failed trial."""

    schema_version: Literal["agentqa.analysis-report/v1alpha1"] = ANALYSIS_REPORT_SCHEMA_VERSION
    analysis_id: NonEmptyString
    generated_at: UtcDateTime
    run_id: NonEmptyString
    test_case_id: ResourceName
    trial_id: NonEmptyString
    trace_id: NonEmptyString
    original_passed: Literal[False]
    evaluation_statuses: dict[ResourceName, NonEmptyString] = Field(max_length=20)
    evidence_hash: Sha256Digest
    evidence_index: list[EvidenceReference] = Field(min_length=1, max_length=100)
    observed_facts: list[ObservedFact] = Field(min_length=1, max_length=50)
    earliest_divergence: TrajectoryDivergence | None = None
    root_causes: list[RootCauseCandidate] = Field(min_length=1, max_length=20)
    missing_evidence: list[AnalysisText] = Field(default_factory=list, max_length=100)
    regression_tests: list[RegressionTestSuggestion] = Field(default_factory=list, max_length=20)
    repair_directions: list[AnalysisText] = Field(default_factory=list, max_length=20)
    llm_call: LLMCallEvidence

    @model_validator(mode="after")
    def validate_references(self) -> "AnalysisReport":
        evidence_ids = [item.evidence_id for item in self.evidence_index]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("analysis evidence ids must be unique")
        known = set(evidence_ids)
        references: list[str] = []
        for fact in self.observed_facts:
            references.extend(fact.evidence_ids)
        if self.earliest_divergence is not None:
            references.extend(self.earliest_divergence.evidence_ids)
        for candidate in self.root_causes:
            references.extend(candidate.evidence_ids)
            references.extend(candidate.counter_evidence_ids)
        for suggestion in self.regression_tests:
            references.extend(suggestion.evidence_ids)
        if set(references) - known:
            raise ValueError("analysis contains unknown evidence ids")
        root_ids = [item.id for item in self.root_causes]
        if len(root_ids) != len(set(root_ids)):
            raise ValueError("root cause candidate ids must be unique")
        if self.llm_call.use_case != "failure_analysis":
            raise ValueError("analysis report requires failure_analysis LLM evidence")
        return self


__all__ = [
    "ANALYSIS_REPORT_SCHEMA_VERSION",
    "AnalysisDraft",
    "AnalysisReport",
    "EvidenceBundle",
    "EvidenceRecord",
    "EvidenceReference",
    "ObservedFact",
    "RegressionTestSuggestion",
    "RootCauseCandidate",
    "TrajectoryDivergence",
]
