"""Transparent callable, tool, and retriever interception wrappers."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

from agentqa.models import InterceptionPoint
from agentqa.mutations.engine import MutationEngine

_IGNORED_ARGUMENTS = {"self", "config", "callbacks", "run_manager"}


def _arguments(
    function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    try:
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return {"args": list(args), "kwargs": kwargs}
    return {
        name: value for name, value in bound.arguments.items() if name not in _IGNORED_ARGUMENTS
    }


def wrap_callable(
    function: Callable[..., Any],
    *,
    engine: MutationEngine,
    interception: InterceptionPoint,
) -> Callable[..., Any]:
    """Wrap a synchronous or asynchronous callable while preserving metadata."""
    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
            return await engine.intercept(
                interception,
                _arguments(function, args, kwargs),
                lambda: function(*args, **kwargs),
            )

        return async_wrapped

    @functools.wraps(function)
    def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
        return engine.intercept_sync(
            interception,
            _arguments(function, args, kwargs),
            lambda: function(*args, **kwargs),
        )

    return sync_wrapped


class _InvocationProxy:
    """Forward object attributes while intercepting common runnable methods."""

    def __init__(
        self,
        wrapped: Any,
        engine: MutationEngine,
        interception: InterceptionPoint,
    ) -> None:
        self._wrapped = wrapped
        self._engine = engine
        self._interception = interception

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        method = self._wrapped
        return self._engine.intercept_sync(
            self._interception,
            _arguments(method, args, kwargs),
            lambda: method(*args, **kwargs),
        )

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        method = self._wrapped.invoke
        return self._engine.intercept_sync(
            self._interception,
            _arguments(method, args, kwargs),
            lambda: method(*args, **kwargs),
        )

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        method = self._wrapped.ainvoke
        return await self._engine.intercept(
            self._interception,
            _arguments(method, args, kwargs),
            lambda: method(*args, **kwargs),
        )

    def get_relevant_documents(self, *args: Any, **kwargs: Any) -> Any:
        method = self._wrapped.get_relevant_documents
        return self._engine.intercept_sync(
            self._interception,
            _arguments(method, args, kwargs),
            lambda: method(*args, **kwargs),
        )

    async def aget_relevant_documents(self, *args: Any, **kwargs: Any) -> Any:
        method = self._wrapped.aget_relevant_documents
        return await self._engine.intercept(
            self._interception,
            _arguments(method, args, kwargs),
            lambda: method(*args, **kwargs),
        )


def _name(value: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    candidate = getattr(value, "name", None) or getattr(value, "__name__", None)
    return str(candidate or type(value).__name__)


def wrap_tool(tool: Any, engine: MutationEngine, *, name: str | None = None) -> Any:
    """Wrap a callable or common LangChain StructuredTool."""
    tool_name = _name(tool, name)
    interception = InterceptionPoint(kind="tool", name=tool_name)
    try:
        module = __import__("langchain_core.tools", fromlist=["StructuredTool"])
        structured_tool = module.StructuredTool
    except (ImportError, AttributeError):
        structured_tool = None
    if structured_tool is not None and isinstance(tool, structured_tool):
        updates: dict[str, Any] = {}
        function = getattr(tool, "func", None)
        coroutine = getattr(tool, "coroutine", None)
        if callable(function):
            updates["func"] = wrap_callable(
                function,
                engine=engine,
                interception=interception,
            )
        if callable(coroutine):
            updates["coroutine"] = wrap_callable(
                coroutine,
                engine=engine,
                interception=interception,
            )
        if updates:
            return tool.model_copy(update=updates)
    if inspect.isfunction(tool) or inspect.ismethod(tool):
        return wrap_callable(tool, engine=engine, interception=interception)
    return _InvocationProxy(tool, engine, interception)


def wrap_retriever(retriever: Any, engine: MutationEngine, *, name: str | None = None) -> Any:
    """Wrap a callable or invoke/ainvoke-compatible retriever."""
    interception = InterceptionPoint(
        kind="retriever",
        name=_name(retriever, name),
    )
    if inspect.isfunction(retriever) or inspect.ismethod(retriever):
        return wrap_callable(retriever, engine=engine, interception=interception)
    return _InvocationProxy(retriever, engine, interception)
