from datetime import UTC, datetime

import httpx
import pytest

from agentqa.models import RunContext
from agentqa.tracing import InMemoryTraceStore, TraceRuntime
from agentqa.tracing.instrumentation import InstrumentationManager
from agentqa.tracing.openai_fallback import OpenAIFallbackInstrumentor

openai = pytest.importorskip("openai")
pytest.importorskip("openinference.instrumentation.openai")


def _context(trace_id: str) -> RunContext:
    return RunContext(
        run_id="run-1",
        spec_hash="spec-hash",
        started_at=datetime.now(UTC),
        random_seed=1,
        test_case_id="case-1",
        trial_id="trial-1",
        trace_id=trace_id,
    )


def _runtime() -> tuple[InMemoryTraceStore, TraceRuntime, InstrumentationManager]:
    store = InMemoryTraceStore()
    runtime = TraceRuntime(store)
    manager = InstrumentationManager()
    manager.activate(["openai"], runtime.provider)
    assert manager.unavailable == []
    return store, runtime, manager


def test_openai_success_span_records_model_tokens_and_no_credentials() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-private"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    store, runtime, manager = _runtime()
    client = openai.OpenAI(
        api_key="sk-private",
        base_url="http://mock.local/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    context = _context("12345678-1234-1234-1234-123456789abc")
    try:
        with runtime.trial_span(context):
            response = client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "user", "content": "sensitive prompt"}],
            )
        assert response.choices[0].message.content == "answer"
        event = next(
            event for event in store.get_events(context.trace_id or "") if event.kind == "LLM"
        )
        assert event.status == "OK"
        assert event.attributes["llm.model_name"] == "gpt-test"
        assert event.attributes["llm.token_count.total"] == 5
        serialized = event.model_dump_json()
        assert "sk-private" not in serialized
        assert "sensitive prompt" not in serialized
    finally:
        client.close()
        manager.close()
        runtime.shutdown()


def test_openai_error_span_is_safe_and_stream_span_ends_after_consumption() -> None:
    stream_body = (
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-test","choices":[{"index":0,"delta":{"content":"done"},'
        '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )

    def error_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            500,
            json={"error": {"message": "private server detail", "type": "server_error"}},
        )

    store, runtime, manager = _runtime()
    error_client = openai.OpenAI(
        api_key="sk-private",
        base_url="http://mock.local/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(error_handler)),
    )
    error_context = _context("22345678-1234-1234-1234-123456789abc")
    try:
        with pytest.raises(openai.InternalServerError), runtime.trial_span(error_context):
            error_client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "user", "content": "private prompt"}],
            )
        error_event = next(
            event for event in store.get_events(error_context.trace_id or "") if event.kind == "LLM"
        )
        assert error_event.status == "ERROR"
        assert "private server detail" not in error_event.model_dump_json()

        stream_client = openai.OpenAI(
            api_key="sk-private",
            base_url="http://mock.local/v1",
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        content=stream_body.encode(),
                    )
                )
            ),
        )
        stream_context = _context("32345678-1234-1234-1234-123456789abc")
        try:
            with runtime.trial_span(stream_context):
                stream = stream_client.chat.completions.create(
                    model="gpt-test",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                )
                assert not store.get_events(stream_context.trace_id or "")
                assert [chunk.choices[0].delta.content for chunk in stream] == ["done"]
                assert any(
                    event.kind == "LLM" for event in store.get_events(stream_context.trace_id or "")
                )
        finally:
            stream_client.close()
    finally:
        error_client.close()
        manager.close()
        runtime.shutdown()


def test_openai_fallback_wrapper_records_safe_summary() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fallback",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "fallback answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        )

    store = InMemoryTraceStore()
    runtime = TraceRuntime(store)
    instrumentor = OpenAIFallbackInstrumentor()
    instrumentor.instrument(tracer_provider=runtime.provider)
    client = openai.OpenAI(
        api_key="sk-private",
        base_url="http://mock.local/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    context = _context("42345678-1234-1234-1234-123456789abc")
    try:
        with runtime.trial_span(context):
            client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "user", "content": "fallback private prompt"}],
            )
        event = next(
            event for event in store.get_events(context.trace_id or "") if event.kind == "LLM"
        )
        assert event.status == "OK"
        assert event.input is not None
        assert event.attributes["llm.token_count.total"] == 5
        assert "fallback private prompt" not in event.model_dump_json()
    finally:
        client.close()
        instrumentor.uninstrument()
        runtime.shutdown()
