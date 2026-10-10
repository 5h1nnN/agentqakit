"""Quality-gate rule evaluation."""

from collections.abc import Callable
from typing import Any

from agentqa.comparison import RunArtifact, metrics_for
from agentqa.errors import AgentQAError
from agentqa.models import (
    ComparisonReport,
    GateResult,
    GateRuleResult,
    QualityGateSpec,
)


class GateError(AgentQAError):
    """A configuration or system error that must produce process exit code 2."""

    def __init__(self, message: str, *, code: str = "GATE_ERROR") -> None:
        super().__init__(message, code=code)


def _rule(
    name: str,
    actual: Any,
    expected: Any,
    passed: bool,
    message: str,
) -> GateRuleResult:
    return GateRuleResult(
        name=name,
        actual=actual,
        expected=expected,
        passed=passed,
        message=message,
    )


def _minimum(name: str, actual: float | None, expected: float) -> GateRuleResult:
    passed = actual is not None and actual >= expected
    message = (
        f"observed {actual:.6g}, required at least {expected:.6g}"
        if actual is not None
        else "configured metric was not observable"
    )
    return _rule(name, actual, expected, passed, message)


def _maximum(name: str, actual: int, expected: int) -> GateRuleResult:
    return _rule(
        name,
        actual,
        expected,
        actual <= expected,
        f"observed {actual}, allowed at most {expected}",
    )


def _relative_maximum(
    name: str,
    comparison: ComparisonReport,
    expected: float,
    select: Callable[[ComparisonReport], Any],
) -> GateRuleResult:
    change = select(comparison)
    if change.status == "available":
        actual = change.relative_increase
        passed = actual is not None and actual <= expected
        message = f"observed {actual:.2%}, allowed at most {expected:.2%}"
        return _rule(name, actual, expected, passed, message)
    actual = {
        "status": change.status,
        "baseline": change.baseline,
        "candidate": change.candidate,
    }
    if change.status == "unbounded":
        message = "candidate increased from a zero baseline; relative increase is unbounded"
    else:
        message = "configured metric was not observable in both runs"
    return _rule(name, actual, expected, False, message)


def evaluate_gate(
    spec: QualityGateSpec,
    candidate: RunArtifact,
    comparison: ComparisonReport | None = None,
) -> GateResult:
    """Evaluate configured rules; missing observations fail instead of becoming zero."""
    rules = spec.rules
    metrics = metrics_for(candidate)
    needs_comparison = (
        rules.max_new_regressions is not None
        or rules.max_p95_latency_increase is not None
        or rules.max_average_cost_increase is not None
    )
    if needs_comparison and comparison is None:
        raise GateError(
            "configured regression rules require --baseline",
            code="GATE_BASELINE_REQUIRED",
        )
    if comparison is not None and comparison.candidate_run_id != candidate.run_id:
        raise GateError(
            "comparison candidate does not match the candidate run",
            code="GATE_INPUT_MISMATCH",
        )

    results: list[GateRuleResult] = []
    if rules.min_pass_rate is not None:
        pass_rate = (
            metrics.pass_rate if comparison is None else comparison.candidate_effective_pass_rate
        )
        results.append(_minimum("minPassRate", pass_rate, rules.min_pass_rate))
    if rules.min_mutation_score is not None:
        results.append(
            _minimum(
                "minMutationScore",
                metrics.mutation_score,
                rules.min_mutation_score,
            )
        )
    if rules.max_new_regressions is not None:
        assert comparison is not None
        results.append(
            _maximum(
                "maxNewRegressions",
                comparison.counts.new_regressions,
                rules.max_new_regressions,
            )
        )
    if rules.max_high_risk_failures is not None:
        results.append(
            _maximum(
                "maxHighRiskFailures",
                metrics.high_risk_failures,
                rules.max_high_risk_failures,
            )
        )
    if rules.max_p95_latency_increase is not None:
        assert comparison is not None
        results.append(
            _relative_maximum(
                "maxP95LatencyIncrease",
                comparison,
                rules.max_p95_latency_increase,
                lambda report: report.deltas.p95_latency_ms,
            )
        )
    if rules.max_average_cost_increase is not None:
        assert comparison is not None
        results.append(
            _relative_maximum(
                "maxAverageCostIncrease",
                comparison,
                rules.max_average_cost_increase,
                lambda report: report.deltas.average_cost_usd,
            )
        )
    if rules.require_replay_bundle_for_failures is not None:
        required = rules.require_replay_bundle_for_failures
        coverage = metrics.replay.failure_bundle_coverage
        passed = not required or coverage == 1.0
        results.append(
            _rule(
                "requireReplayBundleForFailures",
                coverage,
                required,
                passed,
                (
                    "all failed trials have verified replay bundles"
                    if coverage == 1.0
                    else f"verified replay bundle coverage was {coverage:.2%}"
                ),
            )
        )
    passed = all(result.passed for result in results)
    return GateResult(passed=passed, exit_code=0 if passed else 1, rules=results)
