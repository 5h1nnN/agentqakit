from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentqa.models import AgentTestSpec, LangGraphTargetSpec, MutationSpec, RunContext
from agentqa.models import TestInput as TargetInput
from agentqa.runner import RunnerOptions
from agentqa.runner import TestRunner as AgentTestRunner
from agentqa.targets import LangGraphTargetAdapter
from agentqa.tracing import InMemoryTraceStore
from agentqa.tracing.context import reset_current_trace_store, set_current_trace_store
from tests.fixtures.langgraph_targets import configure_replay_tool, replay_tool_calls

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

    store = InMemoryTraceStore()
    token = set_current_trace_store(store)
    try:
        result = await adapter.invoke(
            TargetInput(
                test_case_id="real-graph-case",
                messages=[{"role": "user", "content": "hello"}],
            ),
            context,
        )
    finally:
        reset_current_trace_store(token)
    await adapter.cleanup(context)

    assert result.status == "OK"
    assert result.output == "real-graph-done"
    events = store.get_events(context.trace_id or "")
    assert any(event.kind == "AGENT" for event in events)
    assert any(event.name == "answer" for event in events)


async def test_real_langgraph_tool_boundary_applies_mutation(tmp_path: Path) -> None:
    target = LangGraphTargetSpec.model_validate(
        {
            "metadata": {"name": "real-mutation-graph"},
            "type": "langgraph",
            "entrypoint": "tests.fixtures.real_langgraph_target:build_real_mutation_graph",
        }
    )
    mutation = MutationSpec.model_validate(
        {
            "metadata": {"name": "real-tool-exception"},
            "target": {"kind": "tool", "name": "campus.lookup"},
            "operation": {"type": "tool_exception"},
        }
    )
    test_spec = AgentTestSpec.model_validate(
        {
            "metadata": {"name": "real-mutation-test"},
            "scenario": {"input": {"message": "calendar"}},
            "mutation": "real-tool-exception",
            "expect": {"output": {"requireNonEmpty": True}},
            "evaluators": [{"type": "schema"}],
        }
    ).model_copy(update={"evaluators": []})
    runner = AgentTestRunner(
        LangGraphTargetAdapter(target),
        {},
        options=RunnerOptions(
            artifact_directory=tmp_path / "artifacts",
            trace_flush_timeout_seconds=0.01,
            trace_quiet_period_seconds=0.001,
        ),
        mutations={mutation.metadata.name: mutation},
    )

    result = await runner.run([test_spec], target)

    assert result.results[0].target_result.status == "ERROR"
    assert result.results[0].mutation_applications[0].triggered is True
    assert result.mutation_score is not None
    assert result.mutation_score.overall.killed == 1


async def test_recorded_tool_replays_after_the_real_dependency_is_offline(
    tmp_path: Path,
) -> None:
    target = LangGraphTargetSpec.model_validate(
        {
            "metadata": {"name": "replay-graph"},
            "type": "langgraph",
            "entrypoint": "tests.fixtures.langgraph_targets:build_replay_graph",
        }
    )
    test_spec = AgentTestSpec.model_validate(
        {
            "metadata": {"name": "replay-offline"},
            "scenario": {"input": {"message": "calendar"}},
            "expect": {"output": {"requireNonEmpty": True}},
            "evaluators": [{"type": "schema"}],
        }
    ).model_copy(update={"evaluators": []})
    configure_replay_tool(available=True, reset_calls=True)
    record_runner = AgentTestRunner(
        LangGraphTargetAdapter(target),
        {},
        options=RunnerOptions(
            artifact_directory=tmp_path / "record-artifacts",
            trace_flush_timeout_seconds=0.01,
            trace_quiet_period_seconds=0.001,
        ),
    )

    recorded = await record_runner.run([test_spec], target)

    recorded_trial = recorded.results[0]
    assert recorded_trial.passed is True
    assert recorded_trial.replay is not None
    assert recorded_trial.replay.recorded == 1
    assert replay_tool_calls() == 1
    assert recorded_trial.cassette_ref is not None
    cassette = (
        tmp_path / "record-artifacts" / "runs" / recorded.run_id / recorded_trial.cassette_ref
    )

    configure_replay_tool(available=False)
    replay_runner = AgentTestRunner(
        LangGraphTargetAdapter(target),
        {},
        options=RunnerOptions(
            artifact_directory=tmp_path / "replay-artifacts",
            replay_mode="replay",
            replay_cassette=cassette,
            trace_flush_timeout_seconds=0.01,
            trace_quiet_period_seconds=0.001,
        ),
    )

    replayed = await replay_runner.run([test_spec], target)

    replayed_trial = replayed.results[0]
    assert replayed_trial.passed is True
    assert replayed_trial.target_result.output == recorded_trial.target_result.output
    assert replayed_trial.replay is not None
    assert replayed_trial.replay.hits == 1
    assert replayed_trial.replay.hit_rate == 1.0
    assert replay_tool_calls() == 1
