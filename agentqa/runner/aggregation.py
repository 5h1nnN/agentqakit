"""Deterministic trial and run aggregation."""

from collections.abc import Sequence
from typing import Any

from agentqa.models import AgentTestSpec, MetricSummary, RunSummary, TestCaseSummary, TestResult


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for non-empty values."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_metric(values: Sequence[float]) -> MetricSummary:
    """Compute mean, P50, and P95 with explicit empty handling."""
    if not values:
        return MetricSummary(count=0)
    return MetricSummary(
        count=len(values),
        mean=sum(values) / len(values),
        p50=percentile(values, 0.50),
        p95=percentile(values, 0.95),
    )


def _metadata_number(result: TestResult, key: str) -> float | None:
    value: Any = result.target_result.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _case_passed(spec: AgentTestSpec, pass_rate: float, passed_trials: int) -> bool:
    strategy = spec.trial_policy.strategy
    if strategy == "all":
        return passed_trials == spec.trials
    if strategy == "majority":
        return passed_trials > spec.trials / 2
    threshold = spec.trial_policy.min_pass_rate
    if threshold is None:
        raise ValueError("minPassRate strategy requires a threshold")
    return pass_rate >= threshold


def aggregate_case(spec: AgentTestSpec, results: Sequence[TestResult]) -> TestCaseSummary:
    """Aggregate every completed trial belonging to one test spec."""
    completed = len(results)
    passed_trials = sum(result.passed for result in results)
    failed_trials = completed - passed_trials
    pass_rate = 0.0 if completed == 0 else passed_trials / completed
    complete = completed == spec.trials
    passed = _case_passed(spec, pass_rate, passed_trials) if complete else None
    latencies = [
        (result.target_result.ended_at - result.target_result.started_at).total_seconds() * 1000
        for result in results
    ]
    token_counts = [
        value
        for result in results
        if (value := _metadata_number(result, "total_tokens")) is not None
    ]
    step_counts = [
        value for result in results if (value := _metadata_number(result, "steps")) is not None
    ]
    return TestCaseSummary(
        test_case_id=spec.metadata.name,
        strategy=spec.trial_policy.strategy,
        min_pass_rate=spec.trial_policy.min_pass_rate,
        planned_trials=spec.trials,
        completed_trials=completed,
        passed_trials=passed_trials,
        failed_trials=failed_trials,
        pass_rate=pass_rate,
        passed=passed,
        inconsistent=passed_trials > 0 and failed_trials > 0,
        latency_ms=summarize_metric(latencies),
        total_tokens=summarize_metric(token_counts),
        steps=summarize_metric(step_counts),
    )


def aggregate_run(
    specs: Sequence[AgentTestSpec],
    results: Sequence[TestResult],
) -> RunSummary:
    """Aggregate case decisions while retaining incomplete cancelled work."""
    by_case: dict[str, list[TestResult]] = {spec.metadata.name: [] for spec in specs}
    for result in results:
        if result.test_case_id in by_case:
            by_case[result.test_case_id].append(result)
    cases = [aggregate_case(spec, by_case[spec.metadata.name]) for spec in specs]
    passed = sum(case.passed is True for case in cases)
    failed = sum(case.passed is False for case in cases)
    incomplete = sum(case.passed is None for case in cases)
    total = len(cases)
    return RunSummary(
        total=total,
        passed=passed,
        failed=failed,
        incomplete=incomplete,
        pass_rate=0.0 if total == 0 else passed / total,
        total_trials=sum(spec.trials for spec in specs),
        completed_trials=sum(case.completed_trials for case in cases),
        passed_trials=sum(case.passed_trials for case in cases),
        failed_trials=sum(case.failed_trials for case in cases),
        inconsistent_cases=sum(case.inconsistent for case in cases),
        cases=cases,
    )
