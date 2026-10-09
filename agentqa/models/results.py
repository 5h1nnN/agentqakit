"""Runtime context and result models serialized as snake_case JSON."""

from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from agentqa.models.base import (
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    StrictModel,
    UnitInterval,
    UtcDateTime,
)
from agentqa.models.mutations import MutationApplication, MutationScoreReport


class ErrorInfo(StrictModel):
    """A safe, structured failure without an attached traceback."""

    code: NonEmptyString
    message: NonEmptyString
    details: dict[str, JsonValue] = Field(default_factory=dict)


class RunContext(StrictModel):
    """Immutable identity and reproducibility data for one run."""

    model_config = ConfigDict(frozen=True)

    run_id: NonEmptyString
    spec_hash: NonEmptyString
    started_at: UtcDateTime
    random_seed: int
    manifest_hash: NonEmptyString | None = None
    artifact_dir: NonEmptyString | None = None
    test_case_id: NonEmptyString | None = None
    trial_id: NonEmptyString | None = None
    trace_id: NonEmptyString | None = None
    mutation_id: NonEmptyString | None = None
    trial_seed: int | None = None
    otlp_endpoint: NonEmptyString | None = None
    test_labels: dict[str, str] = Field(default_factory=dict)


class TestInput(StrictModel):
    """Normalized scenario data supplied to every target adapter."""

    test_case_id: NonEmptyString
    input: dict[str, JsonValue] = Field(default_factory=dict)
    initial_state: dict[str, JsonValue] = Field(default_factory=dict)
    messages: list[dict[str, JsonValue]] = Field(default_factory=list)
    mutation: NonEmptyString | None = None


class TargetResult(StrictModel):
    """Normalized output from any target adapter."""

    status: Literal["OK", "ERROR", "TIMEOUT"]
    output: JsonValue = None
    raw_response_ref: NonEmptyString | None = None
    error: ErrorInfo | None = None
    started_at: UtcDateTime
    ended_at: UtcDateTime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_and_error(self) -> "TargetResult":
        if self.ended_at < self.started_at:
            msg = "ended_at must not be before started_at"
            raise ValueError(msg)
        if self.status == "OK" and self.error is not None:
            msg = "successful target result cannot contain an error"
            raise ValueError(msg)
        if self.status != "OK" and self.error is None:
            msg = "failed target result requires an error"
            raise ValueError(msg)
        return self


class TraceEvent(StrictModel):
    """One normalized span-like event in an agent trace."""

    trace_id: NonEmptyString
    span_id: NonEmptyString
    parent_span_id: NonEmptyString | None = None
    kind: Literal["AGENT", "LLM", "TOOL", "RETRIEVER", "NODE", "EVALUATOR"]
    name: NonEmptyString
    input: JsonValue = None
    output: JsonValue = None
    started_at: UtcDateTime
    ended_at: UtcDateTime
    status: Literal["OK", "ERROR", "UNSET"] = "UNSET"
    error: ErrorInfo | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    state_diff: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def validate_time_and_error(self) -> "TraceEvent":
        if self.ended_at < self.started_at:
            msg = "ended_at must not be before started_at"
            raise ValueError(msg)
        if self.status == "ERROR" and self.error is None:
            msg = "ERROR trace event requires error details"
            raise ValueError(msg)
        return self


class TraceRecord(StrictModel):
    """A collected trial trace plus an explicit observability decision."""

    trace_id: NonEmptyString
    status: Literal["complete", "partial", "missing"]
    root_span_id: NonEmptyString | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    diagnostics: list[NonEmptyString] = Field(default_factory=list)
    dropped_spans: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_trace_membership(self) -> "TraceRecord":
        if any(event.trace_id != self.trace_id for event in self.events):
            msg = "every trace event must belong to trace_id"
            raise ValueError(msg)
        if self.status == "missing" and self.root_span_id is not None:
            msg = "missing trace cannot identify a root span"
            raise ValueError(msg)
        if self.status == "complete" and self.root_span_id is None:
            msg = "complete trace requires a root span"
            raise ValueError(msg)
        return self


class Violation(StrictModel):
    """A concrete failed assertion with optional trace evidence."""

    type: NonEmptyString
    message: NonEmptyString
    span_id: NonEmptyString | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationResult(StrictModel):
    """Outcome from one evaluator."""

    evaluator_type: NonEmptyString
    status: Literal["PASS", "FAIL", "ERROR", "SKIPPED", "NOT_OBSERVABLE", "NEEDS_REVIEW"]
    score: UnitInterval | None = None
    message: NonEmptyString | None = None
    violations: list[Violation] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TestResult(StrictModel):
    """Combined target and evaluator result for one test trial."""

    test_case_id: NonEmptyString
    trial_id: NonEmptyString
    trace_id: NonEmptyString
    trial_index: PositiveInt
    trial_seed: int
    started_at: UtcDateTime
    ended_at: UtcDateTime
    passed: bool
    scores: dict[str, UnitInterval] = Field(default_factory=dict)
    violations: list[Violation] = Field(default_factory=list)
    evaluations: list[EvaluationResult] = Field(default_factory=list)
    target_result: TargetResult
    trace_status: Literal["complete", "partial", "missing"] = "missing"
    trace_ref: NonEmptyString | None = None
    mutation_applications: list[MutationApplication] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_time(self) -> "TestResult":
        if self.ended_at < self.started_at:
            msg = "ended_at must not be before started_at"
            raise ValueError(msg)
        return self


class MetricSummary(StrictModel):
    """Deterministic descriptive statistics for one numeric metric."""

    count: NonNegativeInt
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None

    @model_validator(mode="after")
    def validate_empty_state(self) -> "MetricSummary":
        metrics = (self.mean, self.p50, self.p95)
        if self.count == 0 and any(value is not None for value in metrics):
            msg = "empty metric summary cannot contain values"
            raise ValueError(msg)
        if self.count > 0 and any(value is None for value in metrics):
            msg = "non-empty metric summary requires mean, p50, and p95"
            raise ValueError(msg)
        return self


class TestCaseSummary(StrictModel):
    """Trial aggregation and stability metrics for one test case."""

    test_case_id: NonEmptyString
    strategy: Literal["all", "majority", "minPassRate"]
    min_pass_rate: UnitInterval | None = None
    planned_trials: PositiveInt
    completed_trials: NonNegativeInt
    passed_trials: NonNegativeInt
    failed_trials: NonNegativeInt
    pass_rate: UnitInterval
    passed: bool | None
    inconsistent: bool
    latency_ms: MetricSummary
    total_tokens: MetricSummary
    steps: MetricSummary

    @model_validator(mode="after")
    def validate_trial_counts(self) -> "TestCaseSummary":
        if self.passed_trials + self.failed_trials != self.completed_trials:
            msg = "passed_trials plus failed_trials must equal completed_trials"
            raise ValueError(msg)
        if self.completed_trials > self.planned_trials:
            msg = "completed_trials cannot exceed planned_trials"
            raise ValueError(msg)
        expected = 0.0 if self.completed_trials == 0 else self.passed_trials / self.completed_trials
        if abs(self.pass_rate - expected) > 1e-9:
            msg = "pass_rate must equal passed_trials / completed_trials"
            raise ValueError(msg)
        if self.completed_trials < self.planned_trials and self.passed is not None:
            msg = "incomplete test case cannot have a final pass decision"
            raise ValueError(msg)
        return self


class RunSummary(StrictModel):
    """Recomputable aggregate counts for one run."""

    total: NonNegativeInt
    passed: NonNegativeInt
    failed: NonNegativeInt
    incomplete: NonNegativeInt = 0
    pass_rate: UnitInterval
    total_trials: NonNegativeInt = 0
    completed_trials: NonNegativeInt = 0
    passed_trials: NonNegativeInt = 0
    failed_trials: NonNegativeInt = 0
    inconsistent_cases: NonNegativeInt = 0
    cases: list[TestCaseSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "RunSummary":
        if self.passed + self.failed + self.incomplete != self.total:
            msg = "passed plus failed plus incomplete must equal total"
            raise ValueError(msg)
        expected = 0.0 if self.total == 0 else self.passed / self.total
        if abs(self.pass_rate - expected) > 1e-9:
            msg = "pass_rate must equal passed / total"
            raise ValueError(msg)
        if self.passed_trials + self.failed_trials != self.completed_trials:
            msg = "passed_trials plus failed_trials must equal completed_trials"
            raise ValueError(msg)
        if self.completed_trials > self.total_trials:
            msg = "completed_trials cannot exceed total_trials"
            raise ValueError(msg)
        if len(self.cases) != self.total:
            msg = "case summary count must equal total"
            raise ValueError(msg)
        if sum(case.planned_trials for case in self.cases) != self.total_trials:
            msg = "case planned trial counts must equal total_trials"
            raise ValueError(msg)
        if sum(case.completed_trials for case in self.cases) != self.completed_trials:
            msg = "case completed trial counts must equal completed_trials"
            raise ValueError(msg)
        if sum(case.passed_trials for case in self.cases) != self.passed_trials:
            msg = "case passed trial counts must equal passed_trials"
            raise ValueError(msg)
        if sum(case.failed_trials for case in self.cases) != self.failed_trials:
            msg = "case failed trial counts must equal failed_trials"
            raise ValueError(msg)
        if sum(case.inconsistent for case in self.cases) != self.inconsistent_cases:
            msg = "case inconsistency count must equal inconsistent_cases"
            raise ValueError(msg)
        return self


class RunResult(StrictModel):
    """Final state and all trial results for a run."""

    run_id: NonEmptyString
    status: Literal["COMPLETED", "FAILED", "CANCELLED"]
    exit_code: Literal[0, 1, 2]
    started_at: UtcDateTime
    ended_at: UtcDateTime
    summary: RunSummary
    results: list[TestResult] = Field(default_factory=list)
    mutation_score: MutationScoreReport | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def validate_time(self) -> "RunResult":
        if self.ended_at < self.started_at:
            msg = "ended_at must not be before started_at"
            raise ValueError(msg)
        if self.status == "COMPLETED" and self.error is not None:
            msg = "completed run cannot contain a run error"
            raise ValueError(msg)
        if self.status != "COMPLETED" and self.error is None:
            msg = "failed or cancelled run requires a run error"
            raise ValueError(msg)
        expected_exit_code = 2
        if self.status == "COMPLETED" and self.summary.incomplete == 0:
            expected_exit_code = 0 if self.summary.failed == 0 else 1
        if self.exit_code != expected_exit_code:
            msg = "exit_code does not match run status and summary"
            raise ValueError(msg)
        return self


class GateRuleResult(StrictModel):
    """One quality-gate rule decision."""

    name: NonEmptyString
    passed: bool
    actual: JsonValue
    expected: JsonValue
    message: NonEmptyString | None = None


class GateResult(StrictModel):
    """Overall quality-gate decision and process exit code."""

    passed: bool
    exit_code: Literal[0, 1]
    rules: list[GateRuleResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_exit_code(self) -> "GateResult":
        expected = 0 if self.passed else 1
        if self.exit_code != expected:
            msg = "exit_code must be 0 for pass and 1 for fail"
            raise ValueError(msg)
        return self
