"""Task-local trace collection context."""

from contextvars import ContextVar, Token

from agentqa.models import TraceEvent
from agentqa.tracing.options import TraceOptions
from agentqa.tracing.store import TraceStore

_CURRENT_TRACE_STORE: ContextVar[TraceStore | None] = ContextVar(
    "agentqa_current_trace_store",
    default=None,
)
_CURRENT_TRACE_OPTIONS: ContextVar[TraceOptions | None] = ContextVar(
    "agentqa_current_trace_options",
    default=None,
)


def get_current_trace_store() -> TraceStore | None:
    """Return the store associated with the active trial."""
    return _CURRENT_TRACE_STORE.get()


def set_current_trace_store(store: TraceStore) -> Token[TraceStore | None]:
    """Associate *store* with the active task and return a reset token."""
    return _CURRENT_TRACE_STORE.set(store)


def reset_current_trace_store(token: Token[TraceStore | None]) -> None:
    """Restore the preceding task-local trace store."""
    _CURRENT_TRACE_STORE.reset(token)


def get_current_trace_options() -> TraceOptions:
    """Return active privacy limits, or conservative defaults outside a run."""
    return _CURRENT_TRACE_OPTIONS.get() or TraceOptions()


def set_current_trace_options(options: TraceOptions) -> Token[TraceOptions | None]:
    """Associate trace privacy limits with the active task."""
    return _CURRENT_TRACE_OPTIONS.set(options)


def reset_current_trace_options(token: Token[TraceOptions | None]) -> None:
    """Restore the preceding task-local trace options."""
    _CURRENT_TRACE_OPTIONS.reset(token)


def record_trace_event(event: TraceEvent) -> None:
    """Record a framework event when a trial trace store is active."""
    store = get_current_trace_store()
    if store is not None:
        store.add(event)
