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
    QualityCostView,
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


def _improved(change: MetricChange) -> bool | None:
    if change.status == "unobservable" or change.absolute_change is None:
        return None
    return change.absolute_change < 0


def _quality_cost_view(
    *,
    counts: ComparisonCounts,
    deltas: ComparisonDeltas,
) -> QualityCostView:
    quality_preserved = (
        deltas.pass_rate >= 0 and deltas.high_risk_failures <= 0 and counts.new_regressions == 0
    )
    cost_improved = _improved(deltas.average_cost_usd)
    token_improved = _improved(deltas.average_tokens)
    latency_improved = _improved(deltas.p95_latency_ms)
    reasons = [
        f"effective pass-rate change: {deltas.pass_rate:+.6f}",
        f"new regressions: {counts.new_regressions}",
        f"high-risk failure change: {deltas.high_risk_failures:+d}",
    ]
    status: Literal[
        "QUALITY_PRESERVED_COST_IMPROVED",
        "QUALITY_PRESERVED_NO_COST_IMPROVEMENT",
        "QUALITY_COST_TRADEOFF",
        "QUALITY_REGRESSED",
        "COST_UNOBSERVABLE",
    ]
    if cost_improved is None and quality_preserved:
        status = "COST_UNOBSERVABLE"
        reasons.append("average cost is not fully observable")
    elif quality_preserved and cost_improved:
        status = "QUALITY_PRESERVED_COST_IMPROVED"
        reasons.append("average cost decreased without an observed quality regression")
    elif quality_preserved:
        status = "QUALITY_PRESERVED_NO_COST_IMPROVEMENT"
        reasons.append("quality was preserved but average cost did not decrease")
    elif cost_improved:
        status = "QUALITY_COST_TRADEOFF"
        reasons.append("average cost decreased while quality regressed")
    else:
        status = "QUALITY_REGRESSED"
        if cost_improved is None:
            reasons.append("quality regressed and average cost is not fully observable")
        else:
            reasons.append("quality regressed without an average cost improvement")
    return QualityCostView(
        status=status,
        quality_preserved=quality_preserved,
        cost_improved=cost_improved,
        token_improved=token_improved,
        latency_improved=latency_improved,
        reasons=reasons,
    )


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
    deltas = ComparisonDeltas(
        pass_rate=effective_pass_rate - baseline_metrics.pass_rate,
        mutation_score=(
            None
            if baseline_mutation is None or candidate_mutation is None
            else candidate_mutation - baseline_mutation
        ),
        high_risk_failures=(
            candidate_metrics.high_risk_failures - baseline_metrics.high_risk_failures
        ),
        p50_latency_ms=_change(baseline_metrics.latency_ms.p50, candidate_metrics.latency_ms.p50),
        p95_latency_ms=_change(baseline_metrics.latency_ms.p95, candidate_metrics.latency_ms.p95),
        average_tokens=_change(
            baseline_metrics.total_tokens.mean, candidate_metrics.total_tokens.mean
        ),
        average_cost_usd=_change(baseline_metrics.cost_usd.mean, candidate_metrics.cost_usd.mean),
        average_llm_calls=_change(
            baseline_metrics.llm_calls.mean, candidate_metrics.llm_calls.mean
        ),
        average_tool_calls=_change(
            baseline_metrics.tool_calls.mean, candidate_metrics.tool_calls.mean
        ),
        average_retries=_change(baseline_metrics.retries.mean, candidate_metrics.retries.mean),
        average_duplicate_tool_calls=_change(
            baseline_metrics.duplicate_tool_calls.mean,
            candidate_metrics.duplicate_tool_calls.mean,
        ),
        average_no_progress_loops=_change(
            baseline_metrics.no_progress_loops.mean,
            candidate_metrics.no_progress_loops.mean,
        ),
        replay_hit_rate=(
            None
            if baseline_replay is None or candidate_replay is None
            else candidate_replay - baseline_replay
        ),
    )
    return ComparisonReport(
        generated_at=utc_now(),
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        candidate_effective_pass_rate=effective_pass_rate,
        counts=counts,
        deltas=deltas,
        quality_cost=_quality_cost_view(counts=counts, deltas=deltas),
        cases=cases,
    )
