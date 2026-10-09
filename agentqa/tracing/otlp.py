"""OTLP protobuf decoding into AgentQA trace events."""

from typing import Any, cast

from google.protobuf.message import DecodeError  # type: ignore[import-untyped]
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

from agentqa.tracing.normalization import (
    canonical_trace_id,
    nanos_to_datetime,
    normalize_trace_event,
)
from agentqa.tracing.options import TraceOptions
from agentqa.tracing.store import TraceStore


class OTLPDecodeError(ValueError):
    """Raised when an OTLP protobuf payload is malformed."""


def _any_value(value: AnyValue) -> Any:
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "bytes_value":
        return {
            "type": "bytes",
            "length_bytes": len(value.bytes_value),
        }
    if kind == "array_value":
        return [_any_value(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return _attributes(value.kvlist_value.values)
    return None


def _attributes(values: list[KeyValue] | Any) -> dict[str, Any]:
    return {item.key: _any_value(item.value) for item in values}


def _exception_type(span: Any) -> str | None:
    for event in reversed(span.events):
        if event.name != "exception":
            continue
        attributes = _attributes(event.attributes)
        value = attributes.get("exception.type")
        if value is not None:
            return str(value)
    return None


def ingest_otlp_protobuf(
    payload: bytes,
    store: TraceStore,
    options: TraceOptions,
) -> bytes:
    """Decode one ExportTraceServiceRequest and return a protobuf response."""
    request = ExportTraceServiceRequest()
    try:
        request.ParseFromString(payload)
    except DecodeError as exc:
        raise OTLPDecodeError("malformed OTLP protobuf payload") from exc

    for resource_spans in request.resource_spans:
        resource_attributes = {
            f"resource.{key}": value
            for key, value in _attributes(resource_spans.resource.attributes).items()
        }
        for scope_spans in resource_spans.scope_spans:
            scope_attributes: dict[str, Any] = {}
            if scope_spans.scope.name:
                scope_attributes["otel.scope.name"] = scope_spans.scope.name
            if scope_spans.scope.version:
                scope_attributes["otel.scope.version"] = scope_spans.scope.version
            for span in scope_spans.spans:
                if not span.trace_id or not span.span_id or not span.start_time_unix_nano:
                    continue
                attributes = {
                    **resource_attributes,
                    **scope_attributes,
                    **_attributes(span.attributes),
                }
                end_nanos = span.end_time_unix_nano
                if not end_nanos:
                    end_nanos = span.start_time_unix_nano
                    store.add_diagnostic(
                        canonical_trace_id(int.from_bytes(span.trace_id, "big")),
                        "received a span without an end timestamp",
                    )
                status_code = {1: "OK", 2: "ERROR"}.get(span.status.code, "UNSET")
                event = normalize_trace_event(
                    trace_id=int.from_bytes(span.trace_id, "big"),
                    span_id=span.span_id,
                    parent_span_id=span.parent_span_id or None,
                    name=span.name or "unnamed-span",
                    attributes=attributes,
                    started_at=nanos_to_datetime(span.start_time_unix_nano),
                    ended_at=nanos_to_datetime(end_nanos),
                    status_code=status_code,
                    span_kind=span.kind,
                    error_type=_exception_type(span),
                    options=options,
                )
                store.add(event)
    return cast(bytes, ExportTraceServiceResponse().SerializeToString())
