from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from agentqa.models import LangGraphTargetSpec, RunContext
from agentqa.models import TestInput as TargetInput
from agentqa.targets import LangGraphTargetAdapter

pytestmark = pytest.mark.integration
pytest.importorskip("langgraph.graph")


async def test_adapter_runs_real_compiled_langgraph(tmp_path: Path) -> None:
    spec = LangGraphTargetSpec.model_validate(
        {
            "metadata": {"name": "real-graph"},
            "type": "langgraph",
            "entrypoint": "tests.fixtures.real_langgraph_target:build_real_graph",
        }
    )
    context = RunContext(
        run_id="run-1",
        spec_hash="spec-hash",
        started_at=datetime.now(UTC),
        random_seed=1,
        artifact_dir=str(tmp_path),
        test_case_id="real-graph-case",
        trial_id="trial-1",
        trace_id="12345678-1234-1234-1234-123456789abc",
    )
    adapter = LangGraphTargetAdapter(spec)
    await adapter.prepare(context)

    result = await adapter.invoke(
        TargetInput(
            test_case_id="real-graph-case",
            messages=[{"role": "user", "content": "hello"}],
        ),
        context,
    )
    await adapter.cleanup(context)

    assert result.status == "OK"
    assert result.output == "real-graph-done"
    events = cast(list[dict[str, Any]], result.metadata["trace_events"])
    assert any(event["kind"] == "AGENT" for event in events)
    assert any(event["name"] == "answer" for event in events)
