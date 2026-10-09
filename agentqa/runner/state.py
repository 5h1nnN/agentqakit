"""Validated lifecycle state transitions for one test run."""

from dataclasses import dataclass, field
from enum import StrEnum

from agentqa.conventions import utc_now
from agentqa.errors import AgentQAError
from agentqa.models.base import UtcDateTime


class RunState(StrEnum):
    CREATED = "CREATED"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset({RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED})

_ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.PREPARING, RunState.FINALIZING}),
    RunState.PREPARING: frozenset({RunState.RUNNING, RunState.FINALIZING}),
    RunState.RUNNING: frozenset({RunState.EVALUATING, RunState.FINALIZING}),
    RunState.EVALUATING: frozenset({RunState.FINALIZING}),
    RunState.FINALIZING: TERMINAL_STATES,
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
}


class InvalidRunStateTransition(AgentQAError):
    """Raised when code attempts to skip or leave a terminal state."""

    def __init__(self, current: RunState, requested: RunState) -> None:
        super().__init__(
            f"invalid run state transition: {current.value} -> {requested.value}",
            code="RUN_STATE_ERROR",
            details={"current": current.value, "requested": requested.value},
        )


@dataclass(frozen=True, slots=True)
class StateTransition:
    previous: RunState | None
    current: RunState
    at: UtcDateTime


@dataclass(slots=True)
class RunStateMachine:
    """Small explicit finite-state machine for run lifecycle persistence."""

    state: RunState = RunState.CREATED
    history: list[StateTransition] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(StateTransition(previous=None, current=self.state, at=utc_now()))

    def transition(self, requested: RunState) -> None:
        if requested not in _ALLOWED_TRANSITIONS[self.state]:
            raise InvalidRunStateTransition(self.state, requested)
        previous = self.state
        self.state = requested
        self.history.append(StateTransition(previous=previous, current=requested, at=utc_now()))
