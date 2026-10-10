"""Fixtures shared by FileResultStore, SQLite, and API tests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from agentqa.models import (
    AgentTestSpec,
    ErrorInfo,
    ReplayStats,
    RunResult,
    TargetResult,
    TestResult,
    TraceRecord,
    Violation,
)
from agentqa.runner import aggregate_run
from agentqa.storage import ArtifactWriter


def create_storable_run(
    base: Path,
    run_id: str,
    *,
    passed: bool = False,
    failed_lifecycle: bool = False,
) -> tuple[RunResult, Path]:
    """Create one RunResult and matching trace artifact directory."""
    started = datetime(2026, 10, 10, 1, 0, tzinfo=UTC)
    ended = started + timedelta(milliseconds=25)
    spec = AgentTestSpec.model_validate(
        {
            "apiVersion": "agentqa.dev/v1alpha1",
            "kind": "AgentTest",
            "metadata": {"name": "stored-case", "labels": {"risk": "high"}},
            "scenario": {"input": {"message": "hello"}},
            "expect": {"output": {"requireNonEmpty": True}},
            "evaluators": [{"type": "schema"}],
        }
    )
    violations = (
        []
        if passed
        else [
            Violation(
                type="schema",
                message="required field is missing",
                details={"api_key": "finding-secret"},
            )
        ]
    )
    result = TestResult(
        test_case_id=spec.metadata.name,
        trial_id=f"trial-{run_id}",
        trace_id=f"trace-{run_id}",
        trial_index=1,
        trial_seed=42,
        started_at=started,
        ended_at=ended,
        passed=passed,
        violations=violations,
        target_result=TargetResult(
            status="OK",
            output={"answer": "ok", "api_key": "target-secret"},
            started_at=started,
            ended_at=ended,
        ),
        trace_status="missing",
        trace_ref=f"traces/trace-{run_id}.json",
        replay=ReplayStats(
            mode="off",
            recorded=0,
            hits=0,
            misses=0,
            live_fallbacks=0,
        ),
    )
    summary = aggregate_run([spec], [result])
    status: Literal["COMPLETED", "FAILED"] = "FAILED" if failed_lifecycle else "COMPLETED"
    run = RunResult(
        run_id=run_id,
        status=status,
        exit_code=2 if failed_lifecycle else 0 if passed else 1,
        started_at=started,
        ended_at=ended,
        summary=summary,
        results=[result],
        error=(
            ErrorInfo(code="RUN_FAILED", message="fixture lifecycle failed")
            if failed_lifecycle
            else None
        ),
    )
    writer = ArtifactWriter(base, run_id)
    writer.initialize({"run_id": run_id, "status": status})
    writer.write_resolved_tests([spec])
    writer.write_trace(
        result.trace_id,
        TraceRecord(
            trace_id=result.trace_id,
            status="missing",
            diagnostics=["no target-internal spans were observed"],
        ),
    )
    writer.append_result(result)
    writer.finalize_summary(
        {
            "run_id": run_id,
            "status": status,
            "exit_code": run.exit_code,
            "summary": summary.model_dump(mode="json"),
            "mutation_score": None,
        }
    )
    return run, writer.root
