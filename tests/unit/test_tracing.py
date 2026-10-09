import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from opentelemetry.trace import Status, StatusCode

from agentqa.models import RunContext, TraceEvent
from agentqa.tracing import InMemoryTraceStore, JsonlTraceStore, TraceOptions, TraceRuntime


def _context(trace_id: str = "12345678-1234-1234-1234-123456789abc") -> RunContext:
    return RunContext(
        run_id="run-1",
        spec_hash="spec-hash",
        started_at=datetime.now(UTC),
        random_seed=7,
        test_case_id="case-1",
        trial_id="trial-1",
        trace_id=trace_id,
        mutation_id="mutation-1",
    )


async def test_nested_async_spans_keep_trial_parent_and_complete_trace() -> None:
    store = InMemoryTraceStore()
    runtime = TraceRuntime(store)
    context = _context()

    async def nested() -> None:
        await asyncio.sleep(0)
        with runtime.tracer.start_as_current_span(
            "openai.chat.completions",
            attributes={
                "openinference.span.kind": "LLM",
                "input.value": '{"prompt":"private"}',
                "llm.model_name": "test-model",
                "llm.token_count.total": 4,
            },
        ) as child:
            child.set_status(Status(StatusCode.OK))
            await asyncio.sleep(0)

    with runtime.trial_span(context):
        await asyncio.gather(nested(), nested())

    record = store.build_record(context.trace_id or "")
    root = next(event for event in record.events if event.name == "agentqa.test_run")
    children = [event for event in record.events if event.kind == "LLM"]
    assert record.status == "complete"
    assert len(children) == 2
    assert all(event.parent_span_id == root.span_id for event in children)
    assert all(event.attributes["llm.token_count.total"] == 4 for event in children)
    assert all(event.input is not None and "private" not in str(event.input) for event in children)
    runtime.shutdown()


def test_trace_completeness_distinguishes_missing_and_partial() -> None:
    store = InMemoryTraceStore()
    runtime = TraceRuntime(store)
    context = _context()
    with runtime.trial_span(context):
        pass
    missing = store.build_record(context.trace_id or "")
    assert missing.status == "missing"
    assert "no target-internal spans" in missing.diagnostics[0]

    now = datetime.now(UTC)
    store.add(
        TraceEvent(
            trace_id=context.trace_id or "",
            span_id="orphan-span",
            parent_span_id="absent-parent",
            kind="TOOL",
            name="tool",
            started_at=now,
            ended_at=now + timedelta(milliseconds=1),
        )
    )
    partial = store.build_record(context.trace_id or "")
    assert partial.status == "partial"
    assert any("missing parents" in item for item in partial.diagnostics)
    runtime.shutdown()


def test_trace_store_enforces_span_limit_without_raising() -> None:
    store = InMemoryTraceStore(max_spans_per_trace=1)
    now = datetime.now(UTC)
    for index in range(2):
        store.add(
            TraceEvent(
                trace_id="12345678-1234-1234-1234-123456789abc",
                span_id=f"span-{index}",
                kind="NODE",
                name="bounded",
                started_at=now,
                ended_at=now,
            )
        )
    record = store.build_record("12345678-1234-1234-1234-123456789abc")
    assert len(record.events) == 1
    assert record.dropped_spans == 1
    assert "span limit reached" in record.diagnostics


def test_capture_content_is_redacted_and_truncated() -> None:
    from agentqa.tracing.normalization import content_value

    captured = content_value(
        {"authorization": "Bearer private", "prompt": "x" * 200},
        TraceOptions(capture_content=True, max_content_bytes=64),
    )
    assert "Bearer private" not in str(captured)
    assert isinstance(captured, dict)
    assert captured["truncated"] is True


def test_jsonl_trace_store_redacts_credentials_and_skips_dropped_spans(tmp_path: Path) -> None:
    path = tmp_path / "trace-events.jsonl"
    store = JsonlTraceStore(path, max_spans_per_trace=1)
    now = datetime.now(UTC)
    first = TraceEvent(
        trace_id="12345678-1234-1234-1234-123456789abc",
        span_id="span-1",
        kind="NODE",
        name="safe",
        started_at=now,
        ended_at=now,
        attributes={"authorization": "Bearer private"},
    )
    second = first.model_copy(update={"span_id": "span-2"})

    assert store.add(first) is True
    assert store.add(second) is False
    journal = path.read_text(encoding="utf-8")
    assert len(journal.splitlines()) == 1
    assert "Bearer private" not in journal
    assert '"authorization":"***"' in journal
