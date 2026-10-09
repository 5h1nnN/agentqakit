"""Serializable mutation evidence and score models."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from agentqa.models.base import (
    NonEmptyString,
    NonNegativeInt,
    StrictModel,
    UnitInterval,
    UtcDateTime,
)


class InterceptionPoint(StrictModel):
    """A named runtime boundary at which a mutation can be applied."""

    kind: Literal["tool", "retriever", "http"]
    name: NonEmptyString


class MutationMatch(StrictModel):
    """Deterministic condition-by-condition matcher decision."""

    mutation_id: NonEmptyString
    matched: bool
    conditions: dict[str, bool] = Field(default_factory=dict)
    reason: NonEmptyString | None = None
    probability_sample: UnitInterval | None = None


class MutationApplication(StrictModel):
    """Evidence emitted for one active-mutation interception attempt."""

    mutation_id: NonEmptyString
    operation_type: NonEmptyString
    interception: InterceptionPoint
    call_index: int = Field(ge=1)
    triggered: bool
    match: MutationMatch
    invocation_summary: JsonValue
    result_summary: JsonValue = None
    behavior: dict[str, JsonValue] = Field(default_factory=dict)
    test_case_id: NonEmptyString
    trial_id: NonEmptyString
    trace_id: NonEmptyString
    span_id: NonEmptyString
    applied_at: UtcDateTime

    @model_validator(mode="after")
    def validate_decision(self) -> "MutationApplication":
        if self.match.mutation_id != self.mutation_id:
            msg = "match mutation_id must equal application mutation_id"
            raise ValueError(msg)
        if self.triggered and not self.match.matched:
            msg = "a triggered application requires a successful match"
            raise ValueError(msg)
        return self


class MutationOutcome(StrictModel):
    """One mutation's classification across all associated test trials."""

    mutation_id: NonEmptyString
    operation_type: NonEmptyString
    target_kind: NonEmptyString
    status: Literal["KILLED", "SURVIVED", "NOT_REACHED", "INVALID"]
    triggered_count: NonNegativeInt = 0
    test_case_ids: list[NonEmptyString] = Field(default_factory=list)
    failed_test_case_ids: list[NonEmptyString] = Field(default_factory=list)
    reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_classification(self) -> "MutationOutcome":
        if not set(self.failed_test_case_ids).issubset(self.test_case_ids):
            msg = "failed_test_case_ids must be a subset of test_case_ids"
            raise ValueError(msg)
        if self.status == "KILLED" and not self.failed_test_case_ids:
            msg = "KILLED outcome requires a failed test case"
            raise ValueError(msg)
        if self.status != "KILLED" and self.failed_test_case_ids:
            msg = "only KILLED outcomes may contain failed test cases"
            raise ValueError(msg)
        if self.status in {"KILLED", "SURVIVED"} and self.triggered_count == 0:
            msg = f"{self.status} outcome requires a triggered application"
            raise ValueError(msg)
        if self.status == "NOT_REACHED" and self.triggered_count != 0:
            msg = "NOT_REACHED outcome cannot contain a trigger"
            raise ValueError(msg)
        return self


class MutationScoreBucket(StrictModel):
    """Recomputable counts and score for a group of mutation outcomes."""

    killed: NonNegativeInt = 0
    survived: NonNegativeInt = 0
    not_reached: NonNegativeInt = 0
    invalid: NonNegativeInt = 0
    score: UnitInterval | None = None

    @model_validator(mode="after")
    def validate_score(self) -> "MutationScoreBucket":
        denominator = self.killed + self.survived
        expected = None if denominator == 0 else self.killed / denominator
        if self.score != expected:
            msg = "score must equal killed / (killed + survived), or null when empty"
            raise ValueError(msg)
        return self


class MutationScoreReport(StrictModel):
    """Overall and per-operation mutation score report."""

    overall: MutationScoreBucket
    by_operation: dict[str, MutationScoreBucket] = Field(default_factory=dict)
    outcomes: list[MutationOutcome] = Field(default_factory=list)

    @staticmethod
    def _expected_bucket(outcomes: list[MutationOutcome]) -> MutationScoreBucket:
        killed = sum(outcome.status == "KILLED" for outcome in outcomes)
        survived = sum(outcome.status == "SURVIVED" for outcome in outcomes)
        denominator = killed + survived
        return MutationScoreBucket(
            killed=killed,
            survived=survived,
            not_reached=sum(outcome.status == "NOT_REACHED" for outcome in outcomes),
            invalid=sum(outcome.status == "INVALID" for outcome in outcomes),
            score=None if denominator == 0 else killed / denominator,
        )

    @model_validator(mode="after")
    def validate_recomputable_counts(self) -> "MutationScoreReport":
        mutation_ids = [outcome.mutation_id for outcome in self.outcomes]
        if len(mutation_ids) != len(set(mutation_ids)):
            msg = "mutation outcome IDs must be unique"
            raise ValueError(msg)
        if self.overall != self._expected_bucket(self.outcomes):
            msg = "overall bucket must be recomputable from outcomes"
            raise ValueError(msg)
        operation_names = {outcome.operation_type for outcome in self.outcomes}
        if set(self.by_operation) != operation_names:
            msg = "by_operation keys must exactly cover outcome operation types"
            raise ValueError(msg)
        for operation, bucket in self.by_operation.items():
            expected = self._expected_bucket(
                [outcome for outcome in self.outcomes if outcome.operation_type == operation]
            )
            if bucket != expected:
                msg = f"by_operation bucket {operation!r} is not recomputable"
                raise ValueError(msg)
        return self
