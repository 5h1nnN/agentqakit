"""OpenTelemetry runtime used to root and collect each AgentQA trial."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import Span, Status, StatusCode

from agentqa.models import RunContext, TargetResult
from agentqa.tracing.normalization import (
    nanos_to_datetime,
    normalize_trace_event,
    trace_id_hex,
)
from agentqa.tracing.options import TraceOptions
from agentqa.tracing.store import TraceStore

_REQUESTED_TRACE_ID: ContextVar[int | None] = ContextVar(
    "agentqa_requested_otel_trace_id",
    default=None,
)


class _TrialIdGenerator(RandomIdGenerator):
    def generate_trace_id(self) -> int:
        requested = _REQUESTED_TRACE_ID.get()
        return requested if requested is not None else super().generate_trace_id()


def _exception_type(span: ReadableSpan) -> str | None:
    for event in reversed(span.events):
        if event.name != "exception" or event.attributes is None:
            continue
        value = event.attributes.get("exception.type")
        if value is not None:
            return str(value)
    return None


class TraceStoreExporter(SpanExporter):
    """Synchronously normalize ended SDK spans into a bounded TraceStore."""

    def __init__(self, store: TraceStore, options: TraceOptions) -> None:
        self._store = store
        self._options = options

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            try:
                context = span.context
                if context is None or span.start_time is None or span.end_time is None:
                    continue
                parent_id = None if span.parent is None else span.parent.span_id
                attributes: Mapping[str, Any] = span.attributes or {}
                event = normalize_trace_event(
                    trace_id=context.trace_id,
                    span_id=context.span_id,
                    parent_span_id=parent_id,
                    name=span.name,
                    attributes=attributes,
                    started_at=nanos_to_datetime(span.start_time),
                    ended_at=nanos_to_datetime(span.end_time),
                    status_code=span.status.status_code,
                    span_kind=span.kind,
                    error_type=_exception_type(span),
                    options=self._options,
                )
                self._store.add(event)
            except Exception:
                # Instrumentation must never break target execution.
                continue
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


class TraceRuntime:
    """Own an isolated provider so AgentQA never replaces an application's global provider."""

    def __init__(self, store: TraceStore, options: TraceOptions | None = None) -> None:
        self.options = options or TraceOptions()
        self.provider = TracerProvider(
            resource=Resource.create({"service.name": "agentqakit"}),
            id_generator=_TrialIdGenerator(),
            shutdown_on_exit=False,
        )
        self.provider.add_span_processor(
            SimpleSpanProcessor(TraceStoreExporter(store, self.options))
        )
        self.tracer = self.provider.get_tracer("agentqa.runner")

    @contextmanager
    def trial_span(self, context: RunContext) -> Iterator[Span]:
        """Start the canonical root span using the trial's stable trace ID."""
        if context.trace_id is None:
            raise ValueError("trial context requires trace_id")
        token = _REQUESTED_TRACE_ID.set(int(trace_id_hex(context.trace_id), 16))
        attributes: dict[str, str | int | bool] = {
            "agentqa.runner.root": True,
            "agentqa.run.id": context.run_id,
            "agentqa.test.id": context.test_case_id or "unknown",
            "agentqa.trial.id": context.trial_id or "unknown",
        }
        if context.mutation_id is not None:
            attributes["agentqa.mutation.id"] = context.mutation_id
        try:
            with self.tracer.start_as_current_span(
                "agentqa.test_run",
                context=Context(),
                kind=trace.SpanKind.INTERNAL,
                attributes=attributes,
            ) as span:
                yield span
        finally:
            _REQUESTED_TRACE_ID.reset(token)

    @staticmethod
    def finish_target_span(span: Span, result: TargetResult) -> None:
        """Attach bounded outcome metadata and a status before the root span ends."""
        span.set_attribute("agentqa.target.status", result.status)
        latency_ms = (result.ended_at - result.started_at).total_seconds() * 1000
        span.set_attribute("agentqa.target.latency_ms", latency_ms)
        if result.status == "OK":
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR))

    def shutdown(self) -> None:
        """Flush and release this provider's processors."""
        self.provider.shutdown()
