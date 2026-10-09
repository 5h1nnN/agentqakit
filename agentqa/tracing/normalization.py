"""Privacy-aware normalization shared by local and OTLP trace ingestion."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID

from opentelemetry.trace import SpanKind, StatusCode
from pydantic import BaseModel, JsonValue

from agentqa.models import ErrorInfo, TraceEvent
from agentqa.storage import redact_secrets, redact_text_secrets
from agentqa.storage.redaction import is_sensitive_key
from agentqa.tracing.options import TraceOptions

TraceKind = Literal["AGENT", "LLM", "TOOL", "RETRIEVER", "NODE", "EVALUATOR"]

_INPUT_KEYS = (
    "agentqa.input.summary",
    "input.value",
    "input",
    "gen_ai.prompt",
    "gen_ai.input.messages",
    "llm.input_messages",
)
_OUTPUT_KEYS = (
    "agentqa.output.summary",
    "output.value",
    "output",
    "gen_ai.completion",
    "gen_ai.output.messages",
    "llm.output_messages",
)
_LLM_ATTRIBUTE_PARTS = (
    "model",
    "provider",
    "system",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stream",
    "token",
    "finish_reason",
    "response.id",
    "openinference.span.kind",
)
_INVOCATION_PARAMETER_ALLOWLIST = {
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stream",
}


def trace_id_hex(value: str | None) -> str:
    """Return a W3C-compatible 32-hex trace identifier."""
    compact = "" if value is None else value.replace("-", "").lower()
    if (
        len(compact) == 32
        and compact != "0" * 32
        and all(character in "0123456789abcdef" for character in compact)
    ):
        return compact
    return hashlib.sha256((value or "agentqa").encode()).hexdigest()[:32]


def _to_jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 30:
        return "<maximum-depth>"
    if isinstance(value, float) and not math.isfinite(value):
        return "<non-finite-float>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return _to_jsonable(value.model_dump(mode="json"), depth=depth + 1)
    if isinstance(value, Mapping):
        return {
            str(key): _to_jsonable(item, depth=depth + 1)
            for key, item in list(value.items())[:10_000]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item, depth=depth + 1) for item in list(value)[:10_000]]
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"


def canonical_trace_id(value: int | str) -> str:
    """Normalize a W3C trace ID to the UUID representation used by AgentQA."""
    compact = f"{value:032x}" if isinstance(value, int) else trace_id_hex(value)
    return str(UUID(hex=compact))


def span_id_hex(value: int | bytes | str | None) -> str | None:
    """Normalize an OTel span ID to sixteen lowercase hexadecimal characters."""
    if value is None:
        return None
    if isinstance(value, int):
        return f"{value:016x}"
    if isinstance(value, bytes):
        return value.hex().rjust(16, "0")[-16:]
    compact = value.replace("-", "").lower()
    return compact.rjust(16, "0")[-16:]


def nanos_to_datetime(value: int) -> datetime:
    """Convert Unix epoch nanoseconds to an aware UTC datetime."""
    return datetime.fromtimestamp(value / 1_000_000_000, tz=UTC)


def _serialized(value: Any) -> bytes:
    normalized = redact_secrets(_to_jsonable(value))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "array"
    return type(value).__name__


def content_value(value: Any, options: TraceOptions) -> JsonValue:
    """Return either redacted content or a stable non-reversible summary."""
    if value is None:
        return None
    payload = _serialized(value)
    digest = hashlib.sha256(payload).hexdigest()
    if not options.capture_content:
        return {
            "type": _content_type(value),
            "length_bytes": len(payload),
            "sha256": digest,
        }
    if len(payload) <= options.max_content_bytes:
        return cast(JsonValue, redact_secrets(_to_jsonable(value)))
    preview = payload[: options.max_content_bytes].decode("utf-8", errors="replace")
    return {
        "type": _content_type(value),
        "length_bytes": len(payload),
        "sha256": digest,
        "truncated": True,
        "preview": redact_text_secrets(preview),
    }


def _bounded_scalar(value: Any, max_chars: int) -> JsonValue:
    normalized = redact_secrets(_to_jsonable(value))
    if isinstance(normalized, str) and len(normalized) > max_chars:
        return f"{normalized[:max_chars]}…"
    if isinstance(normalized, (dict, list)):
        text = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if len(text) > max_chars:
            return {
                "type": _content_type(normalized),
                "length_chars": len(text),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
    return cast(JsonValue, normalized)


def infer_kind(name: str, attributes: Mapping[str, Any]) -> TraceKind:
    """Map OTel/OpenInference attributes to AgentQA's compact span taxonomy."""
    explicit = str(
        attributes.get("agentqa.span.kind") or attributes.get("openinference.span.kind") or ""
    ).upper()
    if explicit in {"AGENT", "LLM", "TOOL", "RETRIEVER", "NODE", "EVALUATOR"}:
        return cast(TraceKind, explicit)
    lowered = name.lower()
    if any(part in lowered for part in ("chat", "completion", "llm", "openai", "responses")):
        return "LLM"
    if "tool" in lowered:
        return "TOOL"
    if any(part in lowered for part in ("retriev", "vector", "embedding")):
        return "RETRIEVER"
    if attributes.get("agentqa.runner.root") is True:
        return "AGENT"
    return "NODE"


def _content_attribute(key: str) -> bool:
    lowered = key.lower()
    if "token" in lowered:
        return False
    return (
        key in _INPUT_KEYS
        or key in _OUTPUT_KEYS
        or any(
            part in lowered
            for part in (
                "prompt.",
                "completion.",
                "message.content",
                "input.value",
                "output.value",
            )
        )
    )


def sanitize_attributes(
    attributes: Mapping[str, Any],
    *,
    kind: TraceKind,
    options: TraceOptions,
) -> dict[str, JsonValue]:
    """Bound attributes, redact credentials, and whitelist LLM request metadata."""
    result: dict[str, JsonValue] = {}
    for raw_key, value in attributes.items():
        key = str(raw_key)
        lowered = key.lower()
        if is_sensitive_key(key) or any(
            part in lowered for part in ("authorization", "api_key", "apikey", "cookie")
        ):
            result[key] = "***"
        elif _content_attribute(key):
            result[key] = content_value(value, options)
        elif (
            kind != "LLM"
            or key.startswith("agentqa.")
            or any(part in lowered for part in _LLM_ATTRIBUTE_PARTS)
        ):
            result[key] = _bounded_scalar(value, options.max_attribute_chars)
        if len(result) >= options.max_attributes_per_span:
            break
    return result


def extract_content(
    attributes: Mapping[str, Any],
    keys: Sequence[str],
    options: TraceOptions,
) -> JsonValue:
    """Extract the first recognized input/output attribute with privacy handling."""
    for key in keys:
        if key in attributes:
            if key.startswith("agentqa.") and isinstance(attributes[key], str):
                try:
                    return cast(JsonValue, json.loads(attributes[key]))
                except json.JSONDecodeError:
                    pass
            return content_value(attributes[key], options)
    return None


def normalize_trace_event(
    *,
    trace_id: int | str,
    span_id: int | bytes | str,
    parent_span_id: int | bytes | str | None,
    name: str,
    attributes: Mapping[str, Any],
    started_at: datetime,
    ended_at: datetime,
    status_code: StatusCode | int | str,
    span_kind: SpanKind | int | None = None,
    error_type: str | None = None,
    options: TraceOptions,
) -> TraceEvent:
    """Create one validated TraceEvent from normalized OTel values."""
    del span_kind
    normalized_attributes = dict(attributes)
    invocation_parameters = normalized_attributes.pop("llm.invocation_parameters", None)
    if isinstance(invocation_parameters, str):
        try:
            invocation_parameters = json.loads(invocation_parameters)
        except json.JSONDecodeError:
            invocation_parameters = None
    if isinstance(invocation_parameters, Mapping):
        for parameter in _INVOCATION_PARAMETER_ALLOWLIST:
            value = invocation_parameters.get(parameter)
            if isinstance(value, (bool, int, float, str)):
                normalized_attributes[f"llm.invocation_parameters.{parameter}"] = value
    kind = infer_kind(name, normalized_attributes)
    status_text = getattr(status_code, "name", str(status_code)).upper()
    is_error = status_text in {"ERROR", str(StatusCode.ERROR.value)}
    error = None
    if is_error:
        details: dict[str, JsonValue] = {}
        if error_type:
            details["exception_type"] = error_type[: options.max_attribute_chars]
        error = ErrorInfo(
            code="TRACE_SPAN_ERROR",
            message="instrumented operation ended with an error",
            details=details,
        )
    return TraceEvent(
        trace_id=canonical_trace_id(trace_id),
        span_id=cast(str, span_id_hex(span_id)),
        parent_span_id=span_id_hex(parent_span_id),
        kind=kind,
        name=name[: options.max_attribute_chars] or "unnamed-span",
        input=extract_content(normalized_attributes, _INPUT_KEYS, options),
        output=extract_content(normalized_attributes, _OUTPUT_KEYS, options),
        started_at=started_at,
        ended_at=max(ended_at, started_at),
        status="ERROR" if is_error else ("OK" if status_text == "OK" else "UNSET"),
        error=error,
        attributes=sanitize_attributes(normalized_attributes, kind=kind, options=options),
    )
