from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from agentqa.models import LangGraphTargetSpec, RunContext
from agentqa.models import TestInput as TargetInput
from agentqa.targets import LangGraphTargetAdapter


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        run_id="run-1",
        spec_hash="spec-hash",
        started_at=datetime.now(UTC),
        random_seed=7,
        artifact_dir=str(tmp_path),
        test_case_id="case-1",
        trial_id="trial-1",
        trace_id="12345678-1234-1234-1234-123456789abc",
    )


async def test_langgraph_factory_invokes_and_normalizes_events(tmp_path: Path) -> None:
    spec = LangGraphTargetSpec.model_validate(
        {
            "metadata": {"name": "graph"},
            "type": "langgraph",
            "entrypoint": "tests.fixtures.langgraph_targets:build_graph",
            "stateDiffAllowlist": ["safe"],
        }
    )
    adapter = LangGraphTargetAdapter(spec)
    context = _context(tmp_path)
    await adapter.prepare(context)

    result = await adapter.invoke(
        TargetInput(
            test_case_id="case-1",
            messages=[{"role": "user", "content": "hello"}],
        ),
        context,
    )
    await adapter.cleanup(context)

    assert result.status == "OK"
    assert result.output == "done"
    capabilities = cast(dict[str, Any], result.metadata["capabilities"])
    assert capabilities["factory_injection"] is True
    events = cast(list[dict[str, Any]], result.metadata["trace_events"])
    assert isinstance(events, list)
    assert {event["kind"] for event in events} == {"AGENT", "LLM", "TOOL"}
    agent = next(event for event in events if event["kind"] == "AGENT")
    assert agent["state_diff"] == {"safe": 1}


async def test_compiled_graph_reports_injection_hint_and_extracts_path(tmp_path: Path) -> None:
    spec = LangGraphTargetSpec.model_validate(
        {
            "metadata": {"name": "graph"},
            "type": "langgraph",
            "entrypoint": "tests.fixtures.langgraph_targets:compiled_graph",
            "outputPath": "$.answer",
        }
    )
    adapter = LangGraphTargetAdapter(spec)
    context = _context(tmp_path)
    await adapter.prepare(context)

    result = await adapter.invoke(
        TargetInput(test_case_id="case-1", input={"question": "hello"}),
        context,
    )

    assert result.status == "OK"
    assert result.output == "hello"
    capabilities = cast(dict[str, Any], result.metadata["capabilities"])
    assert capabilities["factory_injection"] is False
    assert "build_graph" in capabilities["injection_hint"]
