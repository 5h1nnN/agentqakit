"""Deterministic test-case alignment and run result comparison."""

from typing import Literal

from agentqa.comparison.artifacts import RunArtifact, metrics_for
from agentqa.conventions import utc_now
from agentqa.models import (
    CaseComparison,
    ComparisonCounts,
    ComparisonDeltas,
    ComparisonReport,
    MetricChange,
)


def _change(baseline: float | None, candidate: float | None) -> MetricChange:
    if baseline is None or candidate is None:
        return MetricChange(
            baseline=baseline,
            candidate=candidate,
            status="unobservable",
        )
    absolute = candidate - baseline
    if baseline == 0:
        if candidate == 0:
            return MetricChange(
                baseline=baseline,
                candidate=candidate,
                absolute_change=absolute,
                relative_increase=0.0,
                status="available",
            )
        return MetricChange(
            baseline=baseline,
            candidate=candidate,
            absolute_change=absolute,
            status="unbounded",
        )
    return MetricChange(
        baseline=baseline,
        candidate=candidate,
        absolute_change=absolute,
        relative_increase=absolute / baseline,
        status="available",
    )


def _classification(
    baseline: bool | None, candidate: bool | None
) -> Literal[
    "NEW_REGRESSION",
    "FIXED",
    "PERSISTENT_FAILURE",
    "STABLE",
    "NEW",
    "REMOVED",
]:
    if baseline is None:
        return "NEW"
    if candidate is None:
        return "REMOVED"
    if baseline and not candidate:
        return "NEW_REGRESSION"
    if not baseline and candidate:
        return "FIXED"
    if not baseline and not candidate:
        return "PERSISTENT_FAILURE"
    return "STABLE"


def compare_runs(baseline: RunArtifact, candidate: RunArtifact) -> ComparisonReport:
    """Compare two verified run artifacts by stable test case identifier."""
    baseline_metrics = metrics_for(baseline)
    candidate_metrics = metrics_for(candidate)
    baseline_cases = {case.test_case_id: case.passed for case in baseline.summary.cases}
    candidate_cases = {case.test_case_id: case.passed for case in candidate.summary.cases}
    baseline_risks = {
        test.metadata.name: test.metadata.labels.get("risk") for test in baseline.tests
    }
    candidate_risks = {
        test.metadata.name: test.metadata.labels.get("risk") for test in candidate.tests
    }
    cases = [
        CaseComparison(
            test_case_id=test_case_id,
            classification=_classification(
                baseline_cases.get(test_case_id), candidate_cases.get(test_case_id)
            ),
            baseline_passed=baseline_cases.get(test_case_id),
            candidate_passed=candidate_cases.get(test_case_id),
            risk=(candidate_risks.get(test_case_id) or baseline_risks.get(test_case_id) or None),
        )
        for test_case_id in sorted(set(baseline_cases) | set(candidate_cases))
    ]
    denominator = len(cases)
    effective_pass_rate = (
        0.0
        if denominator == 0
        else sum(case.candidate_passed is True for case in cases) / denominator
    )
    classifications = [case.classification for case in cases]
    counts = ComparisonCounts(
        new_regressions=classifications.count("NEW_REGRESSION"),
        fixed=classifications.count("FIXED"),
        persistent_failures=classifications.count("PERSISTENT_FAILURE"),
        stable=classifications.count("STABLE"),
        new=classifications.count("NEW"),
        removed=classifications.count("REMOVED"),
    )
    baseline_mutation = baseline_metrics.mutation_score
    candidate_mutation = candidate_metrics.mutation_score
    baseline_replay = baseline_metrics.replay.replay_hit_rate
    candidate_replay = candidate_metrics.replay.replay_hit_rate
    return ComparisonReport(
        generated_at=utc_now(),
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        candidate_effective_pass_rate=effective_pass_rate,
        counts=counts,
        deltas=ComparisonDeltas(
            pass_rate=effective_pass_rate - baseline_metrics.pass_rate,
            mutation_score=(
                None
                if baseline_mutation is None or candidate_mutation is None
                else candidate_mutation - baseline_mutation
            ),
            high_risk_failures=(
                candidate_metrics.high_risk_failures - baseline_metrics.high_risk_failures
            ),
            p50_latency_ms=_change(
                baseline_metrics.latency_ms.p50, candidate_metrics.latency_ms.p50
            ),
            p95_latency_ms=_change(
                baseline_metrics.latency_ms.p95, candidate_metrics.latency_ms.p95
            ),
            average_tokens=_change(
                baseline_metrics.total_tokens.mean, candidate_metrics.total_tokens.mean
            ),
            average_cost_usd=_change(
                baseline_metrics.cost_usd.mean, candidate_metrics.cost_usd.mean
            ),
            average_tool_calls=_change(
                baseline_metrics.tool_calls.mean, candidate_metrics.tool_calls.mean
            ),
            replay_hit_rate=(
                None
                if baseline_replay is None or candidate_replay is None
                else candidate_replay - baseline_replay
            ),
        ),
        cases=cases,
    )
