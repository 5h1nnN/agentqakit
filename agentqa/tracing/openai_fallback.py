"""Small OpenAI SDK wrapper used when OpenInference is unavailable."""

import functools
import hashlib
import importlib
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Status, StatusCode

from agentqa.tracing.normalization import content_value
from agentqa.tracing.options import TraceOptions

_REQUEST_PARAMETERS = (
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stream",
)
_PATCH_TARGETS = (
    ("openai.resources.chat.completions", "Completions", False),
    ("openai.resources.chat.completions", "AsyncCompletions", True),
    ("openai.resources.responses.responses", "Responses", False),
    ("openai.resources.responses.responses", "AsyncResponses", True),
)


def _as_jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if value is None or isinstance(value, (bool, int, float, str, list, dict)):
        return value
    return f"<{type(value).__name__}>"


def _summary_attribute(value: Any, options: TraceOptions) -> str:
    return json.dumps(content_value(value, options), sort_keys=True, separators=(",", ":"))


def _set_request_attributes(span: Span, kwargs: dict[str, Any], options: TraceOptions) -> None:
    span.set_attribute("openinference.span.kind", "LLM")
    model = kwargs.get("model")
    if isinstance(model, str):
        span.set_attribute("llm.model_name", model)
    for name in _REQUEST_PARAMETERS:
        value = kwargs.get(name)
        if isinstance(value, (bool, int, float, str)):
            span.set_attribute(f"llm.invocation_parameters.{name}", value)
    request_input = kwargs.get("messages", kwargs.get("input"))
    if request_input is not None:
        span.set_attribute("agentqa.input.summary", _summary_attribute(request_input, options))


def _set_usage_attributes(span: Span, value: Any) -> None:
    usage = getattr(value, "usage", None)
    if usage is None:
        return
    names = {
        "total_tokens": "llm.token_count.total",
        "prompt_tokens": "llm.token_count.prompt",
        "input_tokens": "llm.token_count.prompt",
        "completion_tokens": "llm.token_count.completion",
        "output_tokens": "llm.token_count.completion",
    }
    for source, target in names.items():
        amount = getattr(usage, source, None)
        if isinstance(amount, int):
            span.set_attribute(target, amount)


def _finish_success(span: Span, value: Any, options: TraceOptions) -> None:
    _set_usage_attributes(span, value)
    span.set_attribute("agentqa.output.summary", _summary_attribute(_as_jsonable(value), options))
    span.set_status(Status(StatusCode.OK))
    span.end()


def _finish_error(span: Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR))
    span.end()


class _StreamSummary:
    def __init__(self, span: Span) -> None:
        self._span = span
        self._digest = hashlib.sha256()
        self._length = 0
        self._chunks = 0
        self._ended = False

    def add(self, value: Any) -> None:
        payload = json.dumps(
            _as_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        self._digest.update(payload)
        self._length += len(payload)
        self._chunks += 1
        _set_usage_attributes(self._span, value)

    def success(self) -> None:
        if self._ended:
            return
        self._span.set_attribute(
            "agentqa.output.summary",
            json.dumps(
                {
                    "type": "stream",
                    "chunks": self._chunks,
                    "length_bytes": self._length,
                    "sha256": self._digest.hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
        self._ended = True

    def error(self, exc: BaseException) -> None:
        if not self._ended:
            _finish_error(self._span, exc)
            self._ended = True


class _SyncStreamProxy(Iterator[Any]):
    def __init__(self, wrapped: Any, span: Span) -> None:
        self._wrapped = wrapped
        self._summary = _StreamSummary(span)

    def __iter__(self) -> "_SyncStreamProxy":
        return self

    def __next__(self) -> Any:
        try:
            value = next(self._wrapped)
        except StopIteration:
            self._summary.success()
            raise
        except BaseException as exc:
            self._summary.error(exc)
            raise
        self._summary.add(value)
        return value

    def __enter__(self) -> "_SyncStreamProxy":
        enter = getattr(self._wrapped, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        exit_method = getattr(self._wrapped, "__exit__", None)
        if callable(exit_method):
            exit_method(exc_type, exc, traceback)
        if exc is None:
            self._summary.success()
        else:
            self._summary.error(exc)

    def close(self) -> None:
        close = getattr(self._wrapped, "close", None)
        if callable(close):
            close()
        self._summary.success()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _AsyncStreamProxy(AsyncIterator[Any]):
    def __init__(self, wrapped: Any, span: Span) -> None:
        self._wrapped = wrapped
        self._iterator = wrapped.__aiter__()
        self._summary = _StreamSummary(span)

    def __aiter__(self) -> "_AsyncStreamProxy":
        return self

    async def __anext__(self) -> Any:
        try:
            value = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._summary.success()
            raise
        except BaseException as exc:
            self._summary.error(exc)
            raise
        self._summary.add(value)
        return value

    async def __aenter__(self) -> "_AsyncStreamProxy":
        enter = getattr(self._wrapped, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        exit_method = getattr(self._wrapped, "__aexit__", None)
        if callable(exit_method):
            await exit_method(exc_type, exc, traceback)
        if exc is None:
            self._summary.success()
        else:
            self._summary.error(exc)

    async def close(self) -> None:
        close = getattr(self._wrapped, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result
        self._summary.success()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class OpenAIFallbackInstrumentor:
    """Patch stable OpenAI resource methods with bounded OTel spans."""

    def __init__(self) -> None:
        self._patches: list[tuple[type[Any], Any]] = []

    def instrument(
        self,
        *,
        tracer_provider: TracerProvider,
        trace_options: TraceOptions | None = None,
    ) -> None:
        options = trace_options or TraceOptions()
        tracer = tracer_provider.get_tracer("agentqa.openai.fallback")
        resolved: list[tuple[type[Any], Any, bool]] = []
        for module_name, class_name, is_async in _PATCH_TARGETS:
            module = importlib.import_module(module_name)
            resource_class = cast(type[Any], getattr(module, class_name))
            original = resource_class.create
            resolved.append((resource_class, original, is_async))
        for resource_class, original, is_async in resolved:
            wrapped = (
                self._async_wrapper(original, tracer, options)
                if is_async
                else self._sync_wrapper(original, tracer, options)
            )
            resource_class.create = wrapped
            self._patches.append((resource_class, original))

    @staticmethod
    def _sync_wrapper(original: Any, tracer: Any, options: TraceOptions) -> Any:
        @functools.wraps(original)
        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            span = tracer.start_span("openai.request", kind=trace.SpanKind.CLIENT)
            _set_request_attributes(span, kwargs, options)
            token = otel_context.attach(trace.set_span_in_context(span))
            try:
                result = original(instance, *args, **kwargs)
            except BaseException as exc:
                _finish_error(span, exc)
                raise
            finally:
                otel_context.detach(token)
            if kwargs.get("stream") is True:
                return _SyncStreamProxy(result, span)
            _finish_success(span, result, options)
            return result

        return wrapped

    @staticmethod
    def _async_wrapper(original: Any, tracer: Any, options: TraceOptions) -> Any:
        @functools.wraps(original)
        async def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            span = tracer.start_span("openai.request", kind=trace.SpanKind.CLIENT)
            _set_request_attributes(span, kwargs, options)
            token = otel_context.attach(trace.set_span_in_context(span))
            try:
                result = await original(instance, *args, **kwargs)
            except BaseException as exc:
                _finish_error(span, exc)
                raise
            finally:
                otel_context.detach(token)
            if kwargs.get("stream") is True:
                return _AsyncStreamProxy(result, span)
            _finish_success(span, result, options)
            return result

        return wrapped

    def uninstrument(self) -> None:
        for resource_class, original in reversed(self._patches):
            resource_class.create = original
        self._patches.clear()
