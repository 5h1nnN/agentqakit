"""Task-local mutation execution state."""

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from agentqa.models import MutationApplication, RunContext


@dataclass(slots=True)
class MutationTrialSession:
    """Counters, caches, and evidence isolated to one trial."""

    run_context: RunContext
    counters: dict[tuple[str, str], int] = field(default_factory=dict)
    applied_mutations: set[str] = field(default_factory=set)
    caches: dict[str, Any] = field(default_factory=dict)
    applications: list[MutationApplication] = field(default_factory=list)

    def next_call_index(self, kind: str, name: str) -> int:
        key = (kind, name)
        value = self.counters.get(key, 0) + 1
        self.counters[key] = value
        return value


_CURRENT_MUTATION_SESSION: ContextVar[MutationTrialSession | None] = ContextVar(
    "agentqa_current_mutation_session",
    default=None,
)


def get_current_mutation_session() -> MutationTrialSession | None:
    """Return the active trial's mutation state."""
    return _CURRENT_MUTATION_SESSION.get()


def set_current_mutation_session(
    session: MutationTrialSession,
) -> Token[MutationTrialSession | None]:
    """Activate a mutation session and return its reset token."""
    return _CURRENT_MUTATION_SESSION.set(session)


def reset_current_mutation_session(token: Token[MutationTrialSession | None]) -> None:
    """Restore the preceding mutation session."""
    _CURRENT_MUTATION_SESSION.reset(token)
