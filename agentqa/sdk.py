"""Decorator-based instrumentation for isolated Python agent targets."""

import functools
import inspect
import secrets
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from typing import Any, Literal, ParamSpec, TypeVar, cast, overload

from opentelemetry import trace

from agentqa.conventions import utc_now
from agentqa.models import ErrorInfo, InterceptionPoint, TraceEvent
from agentqa.mutations.context import get_current_mutation_session
from agentqa.mutations.utils import summarize
from agentqa.mutations.wrappers import InvocationInterceptor, wrap_callable
from agentqa.tracing.context import record_trace_event

P = ParamSpec("P")
R = TypeVar("R")

_CURRENT_INTERACTION_ENGINE: ContextVar[InvocationInterceptor | None] = ContextVar(
    "agentqa_sdk_interaction_engine",
    default=None,
)


def _current_run_context() -> Any:
    session = get_current_mutation_session()
    return None if session is None else session.run_context


def _parent_span_id() -> str | None:
    current = trace.get_current_span().get_span_context()
    return f"{current.span_id:016x}" if current.is_valid else None


def _record_call(
    interception: InterceptionPoint,
    *,
    span_id: str,
    parent_span_id: str | None,
    started_at: Any,
    arguments: dict[str, Any],
    result: Any = None,
    error: BaseException | None = None,
) -> None:
    context = _current_run_context()
    if context is None or context.trace_id is None:
        return
    kind: Literal["TOOL", "RETRIEVER"] = "TOOL" if interception.kind == "tool" else "RETRIEVER"
    record_trace_event(
        TraceEvent(
            trace_id=context.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            kind=kind,
            name=f"agentqa.sdk.{interception.kind}.{interception.name}",
            input=summarize(arguments),
            output=None if error is not None else summarize(result),
            started_at=started_at,
            ended_at=utc_now(),
            status="ERROR" if error is not None else "OK",
            error=(
                None
                if error is None
                else ErrorInfo(
                    code="SDK_INTERCEPTED_CALL_ERROR",
                    message="decorated dependency call raised an error",
                    details={"exception_type": type(error).__name__},
                )
            ),
            attributes={
                "agentqa.sdk.decorator": True,
                "agentqa.interception.name": interception.name,
            },
        )
    )


class _SDKInterceptor:
    """Observe every decorated call and delegate active mutation/replay policy."""

    async def intercept(
        self,
        interception: InterceptionPoint,
        arguments: dict[str, Any],
        next_call: Callable[[], Awaitable[Any]],
    ) -> Any:
        started_at = utc_now()
        span_id = secrets.token_hex(8)
        parent_span_id = _parent_span_id()
        engine = _CURRENT_INTERACTION_ENGINE.get()
        try:
            result = (
                await next_call()
                if engine is None
                else await engine.intercept(interception, arguments, next_call)
            )
        except BaseException as exc:
            _record_call(
                interception,
                span_id=span_id,
                parent_span_id=parent_span_id,
                started_at=started_at,
                arguments=arguments,
                error=exc,
            )
            raise
        _record_call(
            interception,
            span_id=span_id,
            parent_span_id=parent_span_id,
            started_at=started_at,
            arguments=arguments,
            result=result,
        )
        return result

    def intercept_sync(
        self,
        interception: InterceptionPoint,
        arguments: dict[str, Any],
        next_call: Callable[[], Any],
    ) -> Any:
        started_at = utc_now()
        span_id = secrets.token_hex(8)
        parent_span_id = _parent_span_id()
        engine = _CURRENT_INTERACTION_ENGINE.get()
        try:
            result = (
                next_call()
                if engine is None
                else engine.intercept_sync(interception, arguments, next_call)
            )
        except BaseException as exc:
            _record_call(
                interception,
                span_id=span_id,
                parent_span_id=parent_span_id,
                started_at=started_at,
                arguments=arguments,
                error=exc,
            )
            raise
        _record_call(
            interception,
            span_id=span_id,
            parent_span_id=parent_span_id,
            started_at=started_at,
            arguments=arguments,
            result=result,
        )
        return result


_SDK_INTERCEPTOR = _SDKInterceptor()


def _decorate_interaction(
    function: Callable[P, R],
    *,
    kind: str,
    name: str | None,
) -> Callable[P, R]:
    logical_name = name or function.__name__
    interception = InterceptionPoint(kind=cast(Any, kind), name=logical_name)
    wrapped = wrap_callable(function, engine=_SDK_INTERCEPTOR, interception=interception)
    metadata_target = cast(Any, wrapped)
    metadata_target.__agentqa_kind__ = kind
    metadata_target.__agentqa_name__ = logical_name
    return cast(Callable[P, R], wrapped)


@overload
def tool(function: Callable[P, R], /, *, name: str | None = None) -> Callable[P, R]: ...


@overload
def tool(
    function: None = None, /, *, name: str | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def tool(
    function: Callable[P, R] | None = None, /, *, name: str | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a sync or async callable as an observable tool boundary."""

    def decorate(value: Callable[P, R]) -> Callable[P, R]:
        return _decorate_interaction(value, kind="tool", name=name)

    return decorate if function is None else decorate(function)


@overload
def retriever(function: Callable[P, R], /, *, name: str | None = None) -> Callable[P, R]: ...


@overload
def retriever(
    function: None = None, /, *, name: str | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def retriever(
    function: Callable[P, R] | None = None, /, *, name: str | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a sync or async callable as an observable retriever boundary."""

    def decorate(value: Callable[P, R]) -> Callable[P, R]:
        return _decorate_interaction(value, kind="retriever", name=name)

    return decorate if function is None else decorate(function)


def _record_agent_call(
    *,
    name: str,
    span_id: str,
    parent_span_id: str | None,
    started_at: Any,
    arguments: dict[str, Any],
    result: Any = None,
    error: BaseException | None = None,
) -> None:
    context = _current_run_context()
    if context is None or context.trace_id is None:
        return
    record_trace_event(
        TraceEvent(
            trace_id=context.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            kind="AGENT",
            name=f"agentqa.sdk.agent.{name}",
            input=summarize(arguments),
            output=None if error is not None else summarize(result),
            started_at=started_at,
            ended_at=utc_now(),
            status="ERROR" if error is not None else "OK",
            error=(
                None
                if error is None
                else ErrorInfo(
                    code="SDK_AGENT_ERROR",
                    message="decorated agent entrypoint raised an error",
                    details={"exception_type": type(error).__name__},
                )
            ),
            attributes={"agentqa.sdk.decorator": True, "agentqa.agent.name": name},
        )
    )


def _decorate_agent(function: Callable[P, R], *, name: str | None) -> Callable[P, R]:
    logical_name = name or function.__name__
    wrapped: Any
    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
            started_at = utc_now()
            span_id = secrets.token_hex(8)
            parent_span_id = _parent_span_id()
            arguments = {"args": list(args), "kwargs": kwargs}
            try:
                result = await function(*args, **kwargs)
            except BaseException as exc:
                _record_agent_call(
                    name=logical_name,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    started_at=started_at,
                    arguments=arguments,
                    error=exc,
                )
                raise
            _record_agent_call(
                name=logical_name,
                span_id=span_id,
                parent_span_id=parent_span_id,
                started_at=started_at,
                arguments=arguments,
                result=result,
            )
            return result

        wrapped = async_wrapped
    else:

        @functools.wraps(function)
        def sync_wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
            started_at = utc_now()
            span_id = secrets.token_hex(8)
            parent_span_id = _parent_span_id()
            arguments = {"args": list(args), "kwargs": kwargs}
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                _record_agent_call(
                    name=logical_name,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    started_at=started_at,
                    arguments=arguments,
                    error=exc,
                )
                raise
            _record_agent_call(
                name=logical_name,
                span_id=span_id,
                parent_span_id=parent_span_id,
                started_at=started_at,
                arguments=arguments,
                result=result,
            )
            return result

        wrapped = sync_wrapped
    metadata_target = cast(Any, wrapped)
    metadata_target.__agentqa_kind__ = "agent"
    metadata_target.__agentqa_name__ = logical_name
    return cast(Callable[P, R], wrapped)


@overload
def agent(function: Callable[P, R], /, *, name: str | None = None) -> Callable[P, R]: ...


@overload
def agent(
    function: None = None, /, *, name: str | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def agent(
    function: Callable[P, R] | None = None, /, *, name: str | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a Python target entrypoint and emit an agent-level trace event."""

    def decorate(value: Callable[P, R]) -> Callable[P, R]:
        return _decorate_agent(value, name=name)

    return decorate if function is None else decorate(function)


def _set_interaction_engine(
    engine: InvocationInterceptor,
) -> Token[InvocationInterceptor | None]:
    return _CURRENT_INTERACTION_ENGINE.set(engine)


def _reset_interaction_engine(token: Token[InvocationInterceptor | None]) -> None:
    _CURRENT_INTERACTION_ENGINE.reset(token)


__all__ = ["agent", "retriever", "tool"]
