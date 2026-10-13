"""Serializable run-comparison models."""

from typing import Final, Literal

from pydantic import Field, model_validator

from agentqa.models.base import (
    NonEmptyString,
    NonNegativeInt,
    StrictModel,
    UnitInterval,
    UtcDateTime,
)
from agentqa.models.results import MetricSummary

COMPARISON_SCHEMA_VERSION: Final = "agentqa.comparison/v1alpha1"


class ReplayCompleteness(StrictModel):
    """Replay and failure-bundle evidence available for one run."""

    total_trials: NonNegativeInt
    replay_trials: NonNegativeInt
    replay_hits: NonNegativeInt
    replay_misses: NonNegativeInt
    live_fallbacks: NonNegativeInt
    replay_hit_rate: UnitInterval | None = None
    failed_trials: NonNegativeInt
    failure_bundles: NonNegativeInt
    failure_bundle_coverage: UnitInterval

    @model_validator(mode="after")
    def validate_rates(self) -> "ReplayCompleteness":
        attempts = self.replay_hits + self.replay_misses
        expected_hit_rate = None if attempts == 0 else self.replay_hits / attempts
        if self.replay_hit_rate != expected_hit_rate:
            msg = "replay_hit_rate must equal hits / (hits + misses), or null when empty"
            raise ValueError(msg)
        expected_coverage = (
            1.0 if self.failed_trials == 0 else self.failure_bundles / self.failed_trials
        )
        if abs(self.failure_bundle_coverage - expected_coverage) > 1e-9:
            msg = "failure_bundle_coverage must equal valid bundles / failed trials"
            raise ValueError(msg)
        return self


class RunMetrics(StrictModel):
    """Comparable quality, performance, cost, and replay metrics for one run."""

    total: NonNegativeInt
    passed: NonNegativeInt
    failed: NonNegativeInt
    pass_rate: UnitInterval
    mutation_score: UnitInterval | None = None
    high_risk_failures: NonNegativeInt
    latency_ms: MetricSummary
    total_tokens: MetricSummary
    cost_usd: MetricSummary
    llm_calls: MetricSummary
    tool_calls: MetricSummary
    retries: MetricSummary
    duplicate_tool_calls: MetricSummary
    no_progress_loops: MetricSummary
    replay: ReplayCompleteness

    @model_validator(mode="after")
    def validate_counts(self) -> "RunMetrics":
        if self.passed + self.failed != self.total:
            msg = "passed plus failed must equal total"
            raise ValueError(msg)
        expected = 0.0 if self.total == 0 else self.passed / self.total
        if abs(self.pass_rate - expected) > 1e-9:
            msg = "pass_rate must equal passed / total"
            raise ValueError(msg)
        return self


class MetricChange(StrictModel):
    """A relative change with explicit unavailable and zero-baseline states."""

    baseline: float | None = None
    candidate: float | None = None
    absolute_change: float | None = None
    relative_increase: float | None = None
    status: Literal["available", "unbounded", "unobservable"]

    @model_validator(mode="after")
    def validate_status(self) -> "MetricChange":
        if self.status == "unobservable":
            if any(
                value is not None
                for value in (
                    self.absolute_change,
                    self.relative_increase,
                )
            ):
                msg = "unobservable metric change cannot contain a calculated change"
                raise ValueError(msg)
            return self
        if self.baseline is None or self.candidate is None or self.absolute_change is None:
            msg = "observable metric change requires baseline, candidate, and absolute_change"
            raise ValueError(msg)
        if self.status == "available" and self.relative_increase is None:
            msg = "available metric change requires relative_increase"
            raise ValueError(msg)
        if self.status == "unbounded" and self.relative_increase is not None:
            msg = "unbounded metric change cannot contain relative_increase"
            raise ValueError(msg)
        return self


class ComparisonDeltas(StrictModel):
    """Candidate-minus-baseline metric changes."""

    pass_rate: float
    mutation_score: float | None = None
    high_risk_failures: int
    p50_latency_ms: MetricChange
    p95_latency_ms: MetricChange
    average_tokens: MetricChange
    average_cost_usd: MetricChange
    average_llm_calls: MetricChange
    average_tool_calls: MetricChange
    average_retries: MetricChange
    average_duplicate_tool_calls: MetricChange
    average_no_progress_loops: MetricChange
    replay_hit_rate: float | None = None


class QualityCostView(StrictModel):
    """Explain quality and efficiency together without collapsing them into one score."""

    status: Literal[
        "QUALITY_PRESERVED_COST_IMPROVED",
        "QUALITY_PRESERVED_NO_COST_IMPROVEMENT",
        "QUALITY_COST_TRADEOFF",
        "QUALITY_REGRESSED",
        "COST_UNOBSERVABLE",
    ]
    quality_preserved: bool
    cost_improved: bool | None
    token_improved: bool | None
    latency_improved: bool | None
    reasons: list[NonEmptyString] = Field(min_length=1, max_length=20)


class CaseComparison(StrictModel):
    """One test case aligned by stable test_case_id."""

    test_case_id: NonEmptyString
    classification: Literal[
        "NEW_REGRESSION",
        "FIXED",
        "PERSISTENT_FAILURE",
        "STABLE",
        "NEW",
        "REMOVED",
    ]
    baseline_passed: bool | None = None
    candidate_passed: bool | None = None
    risk: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_membership(self) -> "CaseComparison":
        if self.classification == "NEW" and (
            self.baseline_passed is not None or self.candidate_passed is None
        ):
            msg = "NEW requires only a candidate outcome"
            raise ValueError(msg)
        if self.classification == "REMOVED" and (
            self.baseline_passed is None or self.candidate_passed is not None
        ):
            msg = "REMOVED requires only a baseline outcome"
            raise ValueError(msg)
        if self.classification not in {"NEW", "REMOVED"} and (
            self.baseline_passed is None or self.candidate_passed is None
        ):
            msg = "aligned comparison requires both outcomes"
            raise ValueError(msg)
        return self


class ComparisonCounts(StrictModel):
    """Counts of case transition classifications."""

    new_regressions: NonNegativeInt
    fixed: NonNegativeInt
    persistent_failures: NonNegativeInt
    stable: NonNegativeInt
    new: NonNegativeInt
    removed: NonNegativeInt


class ComparisonReport(StrictModel):
    """Deterministic comparison between two completed run artifacts."""

    schema_version: Literal["agentqa.comparison/v1alpha1"] = COMPARISON_SCHEMA_VERSION
    generated_at: UtcDateTime
    baseline_run_id: NonEmptyString
    candidate_run_id: NonEmptyString
    baseline: RunMetrics
    candidate: RunMetrics
    candidate_effective_pass_rate: UnitInterval
    counts: ComparisonCounts
    deltas: ComparisonDeltas
    quality_cost: QualityCostView
    cases: list[CaseComparison] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_case_counts(self) -> "ComparisonReport":
        expected = {
            "NEW_REGRESSION": self.counts.new_regressions,
            "FIXED": self.counts.fixed,
            "PERSISTENT_FAILURE": self.counts.persistent_failures,
            "STABLE": self.counts.stable,
            "NEW": self.counts.new,
            "REMOVED": self.counts.removed,
        }
        actual = {
            classification: sum(case.classification == classification for case in self.cases)
            for classification in expected
        }
        if actual != expected:
            msg = "comparison counts must be recomputable from cases"
            raise ValueError(msg)
        denominator = len(self.cases)
        candidate_passed = sum(case.candidate_passed is True for case in self.cases)
        expected_rate = 0.0 if denominator == 0 else candidate_passed / denominator
        if abs(self.candidate_effective_pass_rate - expected_rate) > 1e-9:
            msg = "candidate_effective_pass_rate must retain removed cases in the denominator"
            raise ValueError(msg)
        return self
