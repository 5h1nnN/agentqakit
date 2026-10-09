"""Task-local runner context for future trace instrumentation."""

from contextvars import ContextVar, Token

from agentqa.models import RunContext

_CURRENT_RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar(
    "agentqa_current_run_context",
    default=None,
)


def get_current_run_context() -> RunContext | None:
    """Return the current trial context inside adapter/evaluator execution."""
    return _CURRENT_RUN_CONTEXT.get()


def set_current_run_context(context: RunContext) -> Token[RunContext | None]:
    """Set the current trial context and return a reset token."""
    return _CURRENT_RUN_CONTEXT.set(context)


def reset_current_run_context(token: Token[RunContext | None]) -> None:
    """Restore the context value represented by *token*."""
    _CURRENT_RUN_CONTEXT.reset(token)
