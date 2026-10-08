"""Runtime context and result models serialized as snake_case JSON."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from agentqa.models.base import (
    NonEmptyString,
    NonNegativeInt,
    StrictModel,
    UnitInterval,
    UtcDateTime,
)


class ErrorInfo(StrictModel):
    """A safe, structured failure without an attached traceback."""

    code: NonEmptyString
    message: NonEmptyString
    details: dict[str, JsonValue] = Field(default_factory=dict)


class RunContext(StrictModel):
    """Immutable identity and reproducibility data for one run."""

    run_id: NonEmptyString
    spec_hash: NonEmptyString
    started_at: UtcDateTime
    random_seed: int
    manifest_hash: NonEmptyString | None = None
    artifact_dir: NonEmptyString | None = None


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
    passed: bool
    scores: dict[str, UnitInterval] = Field(default_factory=dict)
    violations: list[Violation] = Field(default_factory=list)
    evaluations: list[EvaluationResult] = Field(default_factory=list)
    target_result: TargetResult | None = None


class RunSummary(StrictModel):
    """Recomputable aggregate counts for one run."""

    total: NonNegativeInt
    passed: NonNegativeInt
    failed: NonNegativeInt
    pass_rate: UnitInterval

    @model_validator(mode="after")
    def validate_counts(self) -> "RunSummary":
        if self.passed + self.failed != self.total:
            msg = "passed plus failed must equal total"
            raise ValueError(msg)
        expected = 0.0 if self.total == 0 else self.passed / self.total
        if abs(self.pass_rate - expected) > 1e-9:
            msg = "pass_rate must equal passed / total"
            raise ValueError(msg)
        return self


class RunResult(StrictModel):
    """Final state and all trial results for a run."""

    run_id: NonEmptyString
    status: Literal["COMPLETED", "FAILED", "CANCELLED"]
    started_at: UtcDateTime
    ended_at: UtcDateTime
    summary: RunSummary
    results: list[TestResult] = Field(default_factory=list)
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def validate_time(self) -> "RunResult":
        if self.ended_at < self.started_at:
            msg = "ended_at must not be before started_at"
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
