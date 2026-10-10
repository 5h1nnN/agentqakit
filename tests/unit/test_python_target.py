from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import agentqa
from agentqa.models import MutationSpec, PythonTargetSpec, RunContext
from agentqa.models import TestInput as TargetInput
from agentqa.mutations import (
    MutationEngine,
    MutationTrialSession,
    reset_current_mutation_session,
    set_current_mutation_session,
)
from agentqa.replay import (
    JsonlCassetteStore,
    ReplaySession,
    reset_current_replay_session,
    set_current_replay_session,
)
from agentqa.targets import PythonTargetAdapter
from agentqa.targets.errors import TargetErrorCode
from agentqa.tracing import InMemoryTraceStore
from agentqa.tracing.context import reset_current_trace_store, set_current_trace_store


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


def _input(value: str = "hello") -> TargetInput:
    return TargetInput(test_case_id="case-1", input={"message": value})


def _spec(
    function: str,
    *,
    timeout_ms: int = 2_000,
    worker_mode: str = "perInvocation",
) -> PythonTargetSpec:
    return PythonTargetSpec.model_validate(
        {
            "metadata": {"name": "python-fixture"},
            "type": "python",
            "entrypoint": f"tests.fixtures.python_targets:{function}",
            "timeoutMs": timeout_ms,
            "workerMode": worker_mode,
        }
    )


@pytest.mark.parametrize("function", ["echo", "async_echo"])
async def test_python_adapter_runs_sync_and_async_callables(function: str, tmp_path: Path) -> None:
    adapter = PythonTargetAdapter(_spec(function))
    context = _context(tmp_path)
    await adapter.prepare(context)

    result = await adapter.invoke(_input(), context)

    await adapter.cleanup(context)
    assert result.status == "OK"
    assert result.output == {"received": {"message": "hello"}}


async def test_python_adapter_keeps_prints_out_of_protocol_and_redacts_log(
    tmp_path: Path,
) -> None:
    adapter = PythonTargetAdapter(_spec("prints"))
    context = _context(tmp_path)
    await adapter.prepare(context)

    result = await adapter.invoke(_input(), context)

    assert result.status == "OK"
    assert result.output == {"message": "hello"}
    assert result.metadata["worker_log"] == "worker user log Authorization: ***\n"


@pytest.mark.parametrize(
    ("function", "code"),
    [
        ("nonserializable", TargetErrorCode.PYTHON_SERIALIZATION_ERROR.value),
        ("raises", TargetErrorCode.PYTHON_EXCEPTION.value),
    ],
)
async def test_python_adapter_normalizes_worker_errors(
    function: str, code: str, tmp_path: Path
) -> None:
    adapter = PythonTargetAdapter(_spec(function))
    context = _context(tmp_path)

    result = await adapter.invoke(_input(), context)

    assert result.status == "ERROR"
    assert result.error is not None
    assert result.error.code == code
    if function == "raises":
        assert result.error.details["exception_type"] == "RuntimeError"
        assert isinstance(result.error.details["stack"], list)


async def test_python_adapter_times_out_and_terminates_exact_worker(tmp_path: Path) -> None:
    adapter = PythonTargetAdapter(_spec("waits_forever", timeout_ms=50))
    context = _context(tmp_path)

    result = await adapter.invoke(_input(), context)

    assert result.status == "TIMEOUT"
    assert result.error is not None
    assert result.error.code == TargetErrorCode.PYTHON_TIMEOUT.value


async def test_per_invocation_mode_prevents_global_state_crosstalk(tmp_path: Path) -> None:
    adapter = PythonTargetAdapter(_spec("counter"))
    context = _context(tmp_path)

    first = await adapter.invoke(_input("one"), context)
    second = await adapter.invoke(_input("two"), context)

    assert first.output == {"calls": 1}
    assert second.output == {"calls": 1}


async def test_persistent_mode_correlates_consecutive_requests(tmp_path: Path) -> None:
    adapter = PythonTargetAdapter(_spec("counter", worker_mode="persistent"))
    context = _context(tmp_path)
    await adapter.prepare(context)

    first = await adapter.invoke(_input("one"), context)
    second = await adapter.invoke(_input("two"), context)
    await adapter.cleanup(context)

    assert first.output == {"calls": 1}
    assert second.output == {"calls": 2}


def test_sdk_decorators_are_transparent_outside_an_agentqa_trial() -> None:
    @agentqa.tool
    def add(left: int, right: int) -> int:
        return left + right

    @agentqa.agent(name="local-agent")
    def run(value: int) -> int:
        return add(value, 2)

    assert run(3) == 5
    assert run.__name__ == "run"
    assert cast(Any, add).__agentqa_kind__ == "tool"
    assert cast(Any, run).__agentqa_name__ == "local-agent"


def _mutation() -> MutationSpec:
    return MutationSpec.model_validate(
        {
            "metadata": {"name": "replace-status"},
            "target": {"kind": "tool", "name": "sdk.lookup"},
            "operation": {"type": "wrong_value", "field": "status", "value": "mutated"},
        }
    )


async def test_python_sdk_bridges_trace_retriever_and_mutation_evidence(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path).model_copy(update={"mutation_id": "replace-status"})
    adapter = PythonTargetAdapter(_spec("decorated_agent"))
    mutation = _mutation()
    adapter.set_mutation_engine(MutationEngine({"replace-status": mutation}))
    mutation_session = MutationTrialSession(context)
    replay_session = ReplaySession(
        run_context=context,
        mode="off",
        store=JsonlCassetteStore(output_path=tmp_path / "sdk.jsonl"),
    )
    trace_store = InMemoryTraceStore()
    mutation_token = set_current_mutation_session(mutation_session)
    replay_token = set_current_replay_session(replay_session)
    trace_token = set_current_trace_store(trace_store)
    try:
        result = await adapter.invoke(_input("course"), context)
    finally:
        reset_current_trace_store(trace_token)
        reset_current_replay_session(replay_token)
        reset_current_mutation_session(mutation_token)

    assert result.status == "OK"
    assert result.output == {
        "message": "course",
        "status": "mutated",
        "documents": [{"page_content": "document for course"}],
    }
    assert len(mutation_session.applications) == 2
    assert sum(application.triggered for application in mutation_session.applications) == 1
    events = trace_store.get_events(context.trace_id or "")
    assert {event.kind for event in events} >= {"AGENT", "TOOL", "RETRIEVER"}
    assert any(event.name == "agentqa.sdk.tool.sdk.lookup" for event in events)
    assert any(event.name == "agentqa.sdk.retriever.sdk.documents" for event in events)


async def test_python_sdk_records_and_strictly_replays_without_live_call(
    tmp_path: Path,
) -> None:
    cassette_path = tmp_path / "recorded.jsonl"
    record_context = _context(tmp_path)
    record_adapter = PythonTargetAdapter(_spec("record_agent"))
    record_mutation = MutationTrialSession(record_context)
    record_session = ReplaySession(
        run_context=record_context,
        mode="record",
        store=JsonlCassetteStore(output_path=cassette_path),
    )
    mutation_token = set_current_mutation_session(record_mutation)
    replay_token = set_current_replay_session(record_session)
    try:
        recorded = await record_adapter.invoke(_input("course"), record_context)
        record_session.store.flush()
    finally:
        reset_current_replay_session(replay_token)
        reset_current_mutation_session(mutation_token)

    assert recorded.status == "OK"
    assert record_session.stats().recorded == 1
    assert cassette_path.is_file()

    replay_context = record_context.model_copy(
        update={"trial_id": "trial-2", "trace_id": "12345678-1234-1234-1234-000000000002"}
    )
    replay_adapter = PythonTargetAdapter(_spec("replay_agent"))
    replay_mutation = MutationTrialSession(replay_context)
    replay_session = ReplaySession(
        run_context=replay_context,
        mode="replay",
        store=JsonlCassetteStore(
            source_path=cassette_path,
            output_path=tmp_path / "replayed.jsonl",
        ),
    )
    mutation_token = set_current_mutation_session(replay_mutation)
    replay_token = set_current_replay_session(replay_session)
    try:
        replayed = await replay_adapter.invoke(_input("course"), replay_context)
    finally:
        reset_current_replay_session(replay_token)
        reset_current_mutation_session(mutation_token)

    assert replayed.status == "OK"
    assert replayed.output == {"query": "course", "source": "live"}
    assert replay_session.stats().hits == 1
    assert replay_session.stats().misses == 0
