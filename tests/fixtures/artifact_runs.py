"""Test helpers for constructing internally consistent run artifact directories."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from agentqa.models import (
    AgentTestSpec,
    MutationOutcome,
    MutationScoreBucket,
    MutationScoreReport,
    ReplayStats,
    TargetResult,
    TestResult,
)
from agentqa.runner.aggregation import aggregate_run
from agentqa.storage import ArtifactWriter


def create_run_artifact(
    base: Path,
    run_id: str,
    cases: dict[str, bool],
    *,
    risks: dict[str, str] | None = None,
    latency_ms: float = 10.0,
    total_tokens: float | None = 100.0,
    cost_usd: float | None = 0.01,
    tool_calls: float | None = 1.0,
    mutation_status: Literal["KILLED", "SURVIVED"] | None = None,
    replay_hits: int = 0,
    replay_misses: int = 0,
    replay_live_fallbacks: int = 0,
) -> Path:
    """Create a checksummed completed run with one trial per test case."""
    risks = risks or {}
    tests = [
        AgentTestSpec.model_validate(
            {
                "apiVersion": "agentqa.dev/v1alpha1",
                "kind": "AgentTest",
                "metadata": {
                    "name": test_case_id,
                    "labels": ({} if test_case_id not in risks else {"risk": risks[test_case_id]}),
                },
                "scenario": {"input": {"message": "hello"}},
                "expect": {"output": {"requireNonEmpty": True}},
                "evaluators": [{"type": "schema"}],
            }
        )
        for test_case_id in cases
    ]
    started_at = datetime(2026, 10, 9, tzinfo=UTC)
    metadata: dict[str, JsonValue] = {
        key: value
        for key, value in {
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "tool_calls": tool_calls,
        }.items()
        if value is not None
    }
    results = [
        TestResult(
            test_case_id=test_case_id,
            trial_id=f"trial-{test_case_id}",
            trace_id=f"trace-{test_case_id}",
            trial_index=1,
            trial_seed=index,
            started_at=started_at,
            ended_at=started_at + timedelta(milliseconds=latency_ms),
            passed=passed,
            target_result=TargetResult(
                status="OK",
                output={"answer": "ok"},
                started_at=started_at,
                ended_at=started_at + timedelta(milliseconds=latency_ms),
                metadata=metadata,
            ),
            replay=ReplayStats(
                mode=("replay" if replay_hits or replay_misses or replay_live_fallbacks else "off"),
                recorded=0,
                hits=replay_hits,
                misses=replay_misses,
                live_fallbacks=replay_live_fallbacks,
                hit_rate=(
                    None
                    if replay_hits + replay_misses == 0
                    else replay_hits / (replay_hits + replay_misses)
                ),
            ),
        )
        for index, (test_case_id, passed) in enumerate(cases.items(), start=1)
    ]
    summary = aggregate_run(tests, results)
    mutation_score = None
    if mutation_status is not None:
        killed = int(mutation_status == "KILLED")
        survived = int(mutation_status == "SURVIVED")
        bucket = MutationScoreBucket(
            killed=killed,
            survived=survived,
            score=float(killed),
        )
        outcome = MutationOutcome(
            mutation_id="fixture-mutation",
            operation_type="tool_exception",
            target_kind="tool",
            status=mutation_status,
            triggered_count=1,
            test_case_ids=[next(iter(cases))],
            failed_test_case_ids=([next(iter(cases))] if mutation_status == "KILLED" else []),
        )
        mutation_score = MutationScoreReport(
            overall=bucket,
            by_operation={"tool_exception": bucket},
            outcomes=[outcome],
        )
    writer = ArtifactWriter(base, run_id)
    writer.initialize({"run_id": run_id, "status": "COMPLETED"})
    writer.write_resolved_tests(tests)
    for result in results:
        writer.append_result(result)
    writer.write_run({"run_id": run_id, "status": "COMPLETED"})
    if mutation_score is not None:
        writer.write_mutation_score(mutation_score)
    writer.finalize_summary(
        {
            "run_id": run_id,
            "status": "COMPLETED",
            "exit_code": 0 if summary.failed == 0 else 1,
            "summary": summary.model_dump(mode="json"),
            "mutation_score": (
                None if mutation_score is None else mutation_score.model_dump(mode="json")
            ),
        }
    )
    return writer.root
