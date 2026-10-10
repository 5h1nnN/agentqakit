import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("langgraph.graph")

from examples.agents.campus_langgraph import (  # noqa: E402
    build_v0_bug1,
    build_v0_bug2,
    build_v0_bug3,
    build_v0_bug4,
    build_v1_fixed,
)
from examples.agents.opensource_langgraph import build_graph  # noqa: E402


class _PassthroughRuntime:
    def wrap_tool(self, function: Any, *, name: str) -> Any:
        del name
        return function

    def wrap_retriever(self, function: Any, *, name: str) -> Any:
        del name
        return function


async def _invoke(factory: Any, state: dict[str, Any]) -> dict[str, Any]:
    graph = factory(agentqa_runtime=_PassthroughRuntime())
    return cast(dict[str, Any], await graph.ainvoke(state))


async def test_campus_fixed_and_defect_variants_are_reproducible() -> None:
    missing = {"action": "register_event", "user_id": "student-001"}
    assert (await _invoke(build_v1_fixed, missing))["result"]["status"] == "invalid_request"
    assert (await _invoke(build_v0_bug2, missing))["result"]["status"] == "registered"

    cross_user = {
        "action": "query_score",
        "user_id": "student-001",
        "subject_user_id": "student-002",
        "course_id": "AI101",
    }
    assert (await _invoke(build_v1_fixed, cross_user))["result"]["status"] == "refused"
    assert (await _invoke(build_v0_bug3, cross_user))["result"]["status"] == "ok"

    retrieval = {"action": "retrieve_policy", "message": "活动报名政策"}
    assert (await _invoke(build_v1_fixed, retrieval))["result"]["status"] == "safe_summary"
    assert (await _invoke(build_v0_bug4, retrieval))["result"]["status"] == "unsafe"

    timeout = {
        "action": "query_course",
        "course_id": "AI101",
        "simulate_timeout": True,
    }
    fixed = await _invoke(build_v1_fixed, timeout)
    assert fixed["result"]["status"] == "temporary_failure"
    assert fixed["result"]["retries"] == 2
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_invoke(build_v0_bug1, timeout), timeout=0.05)


async def test_opensource_agent_keeps_test_execution_allowlisted() -> None:
    refused = await _invoke(build_graph, {"action": "run_tests", "command": "rm -rf repo"})
    allowed = await _invoke(
        build_graph,
        {"action": "run_tests", "command": "python -m pytest -q"},
    )

    assert refused["result"]["status"] == "refused"
    assert allowed["result"]["status"] == "passed"


def test_benchmark_smoke_produces_two_real_results(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    output = tmp_path / "results"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_benchmark.py"),
            "--config",
            str(repository / "benchmark" / "config.yaml"),
            "--output",
            str(output),
            "--configurations",
            "output-only",
            "--limit",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    rows = [
        json.loads(line)
        for line in (output / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert {row["suite"] for row in rows} == {"campus", "opensource"}
    assert all(row["passed"] for row in rows)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["resultCount"] == 2
    for relative in (
        "metrics.csv",
        "mutation_matrix.csv",
        "environment.json",
        "charts/pass-rate.svg",
    ):
        assert (output / relative).is_file()
