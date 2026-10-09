from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from agentqa.tracing import InMemoryTraceStore, LocalOTLPReceiver, TraceOptions
from agentqa.tracing.normalization import canonical_trace_id

pytestmark = pytest.mark.integration


def test_standard_otlp_exporter_submits_queryable_span_and_bad_payload_is_400() -> None:
    store = InMemoryTraceStore()
    try:
        receiver = LocalOTLPReceiver(store, port=0, options=TraceOptions())
    except PermissionError:
        pytest.skip("the execution sandbox does not permit loopback socket binding")
    receiver.start()
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = OTLPSpanExporter(endpoint=f"{receiver.endpoint}/v1/traces")
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("agentqa-test")
    try:
        with tracer.start_as_current_span(
            "openai.chat.completions",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "openinference.span.kind": "LLM",
                "llm.model_name": "mock-model",
                "llm.token_count.total": 7,
            },
        ) as span:
            trace_id = canonical_trace_id(span.get_span_context().trace_id)
        assert provider.force_flush(timeout_millis=2_000)
        events = store.get_events(trace_id)
        assert len(events) == 1
        assert events[0].attributes["llm.token_count.total"] == 7

        request = Request(
            f"{receiver.endpoint}/v1/traces",
            data=b"not-protobuf",
            headers={"content-type": "application/x-protobuf"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2)
        assert error.value.code == 400
    finally:
        provider.shutdown()
        receiver.stop()
