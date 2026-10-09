import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agentqa.models import MutationSpec, RunContext
from agentqa.mutations import (
    MutationEngine,
    MutationTrialSession,
    reset_current_mutation_session,
    set_current_mutation_session,
    wrap_tool,
)
from agentqa.replay import (
    CassetteFormatError,
    CassetteMissError,
    InteractionEngine,
    JsonlCassetteStore,
    ReplayedDependencyError,
    ReplaySession,
    reset_current_replay_session,
    set_current_replay_session,
)
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.tracing import InMemoryTraceStore
from agentqa.tracing.context import reset_current_trace_store, set_current_trace_store


def _context() -> RunContext:
    return RunContext(
        run_id="run-1",
        spec_hash="spec-1",
        started_at=datetime.now(UTC),
        random_seed=7,
        test_case_id="case-1",
        trial_id="trial-1",
        trace_id="trace-1",
        trial_seed=11,
    )


def _activate(session: ReplaySession) -> tuple[Any, Any, InMemoryTraceStore]:
    traces = InMemoryTraceStore()
    replay_token = set_current_replay_session(session)
    trace_token = set_current_trace_store(traces)
    return replay_token, trace_token, traces


def _deactivate(replay_token: Any, trace_token: Any) -> None:
    reset_current_trace_store(trace_token)
    reset_current_replay_session(replay_token)


def test_record_and_replay_are_canonical_redacted_and_call_indexed(tmp_path: Path) -> None:
    recorded_path = tmp_path / "recorded.jsonl"
    recorded_store = JsonlCassetteStore(output_path=recorded_path)
    recorded_session = ReplaySession(
        run_context=_context(),
        mode="record",
        store=recorded_store,
        volatile_fields=("nonce",),
    )
    replay_token, trace_token, traces = _activate(recorded_session)
    calls = 0

    def lookup(query: str, nonce: str, api_key: str) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"answer": query, "authorization": f"Bearer {api_key}"}

    wrapped = wrap_tool(lookup, InteractionEngine(), name="campus.lookup")
    try:
        assert wrapped("calendar", "first", "top-secret")["answer"] == "calendar"
        assert wrapped("calendar", "second", "top-secret")["answer"] == "calendar"
    finally:
        recorded_store.flush()
        _deactivate(replay_token, trace_token)

    assert calls == 2
    assert [entry.call_index for entry in recorded_store.recorded_entries] == [1, 2]
    assert all(
        entry.arguments == {"api_key": "***", "query": "calendar"}
        for entry in recorded_store.recorded_entries
    )
    assert "top-secret" not in recorded_path.read_text(encoding="utf-8")
    lines = recorded_path.read_bytes().splitlines()
    assert all(line == canonical_json_bytes(json.loads(line)) for line in lines)
    assert all(entry.verify_checksum() for entry in recorded_store.recorded_entries)
    assert {event.span_id for event in traces.get_events("trace-1")} == {
        entry.span_id for entry in recorded_store.recorded_entries
    }

    replay_output = tmp_path / "replayed.jsonl"
    replay_store = JsonlCassetteStore(
        source_path=recorded_path,
        output_path=replay_output,
    )
    replay_session = ReplaySession(
        run_context=_context(),
        mode="replay",
        store=replay_store,
        volatile_fields=("nonce",),
    )
    replay_token, trace_token, _ = _activate(replay_session)

    def unavailable(query: str, nonce: str, api_key: str) -> dict[str, str]:
        del query, nonce, api_key
        raise AssertionError("the live dependency must remain offline")

    replayed = wrap_tool(unavailable, InteractionEngine(), name="campus.lookup")
    try:
        assert replayed("calendar", "new-a", "different-secret")["answer"] == "calendar"
        assert replayed("calendar", "new-b", "different-secret")["answer"] == "calendar"
    finally:
        replay_store.flush()
        _deactivate(replay_token, trace_token)

    assert replay_session.stats().hits == 2
    assert replay_session.stats().misses == 0
    assert replay_session.stats().hit_rate == 1.0
    assert len(replay_output.read_text(encoding="utf-8").splitlines()) == 2


async def test_recorded_dependency_errors_are_replayed_without_live_access(
    tmp_path: Path,
) -> None:
    cassette = tmp_path / "errors.jsonl"
    record_store = JsonlCassetteStore(output_path=cassette)
    record_session = ReplaySession(run_context=_context(), mode="record", store=record_store)
    replay_token, trace_token, _ = _activate(record_session)

    async def failing_tool(query: str) -> None:
        del query
        raise ValueError("password=do-not-persist")

    wrapped = wrap_tool(failing_tool, InteractionEngine(), name="campus.failure")
    try:
        with pytest.raises(ValueError):
            await wrapped("test")
    finally:
        record_store.flush()
        _deactivate(replay_token, trace_token)

    assert "do-not-persist" not in cassette.read_text(encoding="utf-8")
    entry = record_store.recorded_entries[0]
    assert entry.error is not None
    assert entry.error.exception_type == "ValueError"

    replay_store = JsonlCassetteStore(
        source_path=cassette,
        output_path=tmp_path / "replayed-errors.jsonl",
    )
    replay_session = ReplaySession(run_context=_context(), mode="replay", store=replay_store)
    replay_token, trace_token, _ = _activate(replay_session)
    live_calls = 0

    async def offline_tool(query: str) -> None:
        nonlocal live_calls
        del query
        live_calls += 1

    wrapped = wrap_tool(offline_tool, InteractionEngine(), name="campus.failure")
    try:
        with pytest.raises(ReplayedDependencyError) as error:
            await wrapped("test")
    finally:
        replay_store.flush()
        _deactivate(replay_token, trace_token)

    assert error.value.original_exception_type == "ValueError"
    assert live_calls == 0
    assert replay_session.hits == 1


def test_replay_miss_fails_closed_unless_live_fallback_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")

    strict_store = JsonlCassetteStore(
        source_path=source,
        output_path=tmp_path / "strict.jsonl",
    )
    strict = ReplaySession(run_context=_context(), mode="replay", store=strict_store)
    replay_token, trace_token, _ = _activate(strict)
    live_calls = 0

    def dependency(value: int) -> int:
        nonlocal live_calls
        live_calls += 1
        return value + 1

    wrapped = wrap_tool(dependency, InteractionEngine(), name="campus.missing")
    try:
        with pytest.raises(CassetteMissError):
            wrapped(1)
    finally:
        strict_store.flush()
        _deactivate(replay_token, trace_token)
    assert live_calls == 0
    assert strict.misses == 1

    fallback_store = JsonlCassetteStore(
        source_path=source,
        output_path=tmp_path / "fallback.jsonl",
    )
    fallback = ReplaySession(
        run_context=_context(),
        mode="replay",
        store=fallback_store,
        allow_live_fallback=True,
    )
    replay_token, trace_token, _ = _activate(fallback)
    wrapped = wrap_tool(dependency, InteractionEngine(), name="campus.missing")
    try:
        assert wrapped(1) == 2
    finally:
        fallback_store.flush()
        _deactivate(replay_token, trace_token)
    assert live_calls == 1
    assert fallback.misses == 1
    assert fallback.live_fallbacks == 1
    assert len(fallback_store.recorded_entries) == 1


def test_large_results_use_verified_blob_references(tmp_path: Path) -> None:
    cassette = tmp_path / "large.jsonl"
    store = JsonlCassetteStore(output_path=cassette, max_inline_result_bytes=16)
    result: Any = {"text": "x" * 100}
    entry = store.record_success(
        kind="retriever",
        name="campus.docs",
        arguments={"query": "calendar"},
        call_index=1,
        result=result,
        trace_id="trace-1",
        span_id="span-1",
    )
    store.flush()

    assert entry.result is None
    assert entry.result_ref is not None
    blob = cassette.parent / entry.result_ref
    assert blob.is_file()
    assert blob.with_name(f"{blob.name}.sha256").read_text(encoding="utf-8").strip() == (
        sha256_digest(blob.read_bytes())
    )
    assert store.result(entry) == result


def test_mutations_are_applied_outside_the_recorded_baseline(tmp_path: Path) -> None:
    mutation = MutationSpec.model_validate(
        {
            "metadata": {"name": "wrong-answer"},
            "target": {"kind": "tool", "name": "campus.lookup"},
            "operation": {"type": "wrong_value", "field": "answer", "value": "mutated"},
        }
    )
    context = _context().model_copy(update={"mutation_id": mutation.metadata.name})
    cassette = JsonlCassetteStore(output_path=tmp_path / "baseline.jsonl")
    replay_session = ReplaySession(run_context=context, mode="record", store=cassette)
    replay_token, trace_token, _ = _activate(replay_session)
    mutation_session = MutationTrialSession(context)
    mutation_token = set_current_mutation_session(mutation_session)
    wrapped = wrap_tool(
        lambda: {"answer": "baseline"},
        InteractionEngine(MutationEngine({mutation.metadata.name: mutation})),
        name="campus.lookup",
    )
    try:
        assert wrapped() == {"answer": "mutated"}
    finally:
        cassette.flush()
        reset_current_mutation_session(mutation_token)
        _deactivate(replay_token, trace_token)

    assert cassette.recorded_entries[0].result == {"answer": "baseline"}
    assert mutation_session.applications[0].triggered is True


def test_source_cassette_rejects_tampered_entry_checksums(tmp_path: Path) -> None:
    cassette = tmp_path / "source.jsonl"
    store = JsonlCassetteStore(output_path=cassette)
    store.record_success(
        kind="tool",
        name="campus.lookup",
        arguments={"query": "calendar"},
        call_index=1,
        result="available",
        trace_id="trace-1",
        span_id="span-1",
    )
    store.flush()
    cassette.write_text(
        cassette.read_text(encoding="utf-8").replace("calendar", "tampered"),
        encoding="utf-8",
    )

    with pytest.raises(CassetteFormatError, match="checksum mismatch"):
        JsonlCassetteStore(source_path=cassette)
    with pytest.raises(ValueError, match="must differ"):
        JsonlCassetteStore(source_path=cassette, output_path=cassette)
