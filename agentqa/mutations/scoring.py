"""Recomputable mutation outcome classification and scoring."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from agentqa.models import (
    AgentTestSpec,
    MutationOutcome,
    MutationScoreBucket,
    MutationScoreReport,
    MutationSpec,
    TestResult,
)


def _bucket(outcomes: Sequence[MutationOutcome]) -> MutationScoreBucket:
    killed = sum(item.status == "KILLED" for item in outcomes)
    survived = sum(item.status == "SURVIVED" for item in outcomes)
    denominator = killed + survived
    return MutationScoreBucket(
        killed=killed,
        survived=survived,
        not_reached=sum(item.status == "NOT_REACHED" for item in outcomes),
        invalid=sum(item.status == "INVALID" for item in outcomes),
        score=None if denominator == 0 else killed / denominator,
    )


def _classify(
    mutation: MutationSpec,
    test_specs: Sequence[AgentTestSpec],
    results: Sequence[TestResult],
) -> MutationOutcome:
    mutation_id = mutation.metadata.name
    associated = [spec for spec in test_specs if spec.mutation == mutation_id]
    test_case_ids = [spec.metadata.name for spec in associated]
    relevant = [result for result in results if result.test_case_id in set(test_case_ids)]
    triggered_count = sum(
        application.triggered
        for result in relevant
        for application in result.mutation_applications
        if application.mutation_id == mutation_id
    )

    def outcome(
        status: Literal["KILLED", "SURVIVED", "NOT_REACHED", "INVALID"],
        *,
        reason: str | None = None,
        failed_test_case_ids: list[str] | None = None,
    ) -> MutationOutcome:
        return MutationOutcome(
            mutation_id=mutation_id,
            operation_type=mutation.operation.type,
            target_kind=mutation.target.kind,
            status=status,
            triggered_count=triggered_count,
            test_case_ids=test_case_ids,
            failed_test_case_ids=failed_test_case_ids or [],
            reason=reason,
        )

    if not associated:
        return outcome("INVALID", reason="mutation has no associated AgentTest")
    expected_trials = sum(spec.trials for spec in associated)
    if len(relevant) != expected_trials:
        return outcome("INVALID", reason="one or more associated trials did not complete")
    if triggered_count == 0:
        return outcome("NOT_REACHED", reason="no MutationApplication reported a trigger")

    failed = [
        result.test_case_id
        for result in relevant
        if result.target_result.status != "OK"
        or any(evaluation.status == "FAIL" for evaluation in result.evaluations)
    ]
    if failed:
        return outcome("KILLED", failed_test_case_ids=list(dict.fromkeys(failed)))
    inconclusive = any(
        evaluation.status in {"ERROR", "SKIPPED", "NOT_OBSERVABLE", "NEEDS_REVIEW"}
        for result in relevant
        for evaluation in result.evaluations
    )
    if inconclusive or any(not result.evaluations for result in relevant):
        return outcome("INVALID", reason="associated evaluations were missing or inconclusive")
    return outcome("SURVIVED")


def calculate_mutation_score(
    mutations: Mapping[str, MutationSpec],
    test_specs: Sequence[AgentTestSpec],
    results: Sequence[TestResult],
) -> MutationScoreReport:
    """Classify every configured mutation and compute honest score denominators."""
    outcomes = [
        _classify(mutation, test_specs, results) for _, mutation in sorted(mutations.items())
    ]
    grouped: dict[str, list[MutationOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.operation_type].append(outcome)
    return MutationScoreReport(
        overall=_bucket(outcomes),
        by_operation={name: _bucket(values) for name, values in sorted(grouped.items())},
        outcomes=outcomes,
    )
