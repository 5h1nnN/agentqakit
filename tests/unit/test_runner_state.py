import pytest

from agentqa.runner import InvalidRunStateTransition, RunState, RunStateMachine


def test_run_state_machine_accepts_the_complete_lifecycle() -> None:
    machine = RunStateMachine()

    for state in (
        RunState.PREPARING,
        RunState.RUNNING,
        RunState.EVALUATING,
        RunState.FINALIZING,
        RunState.COMPLETED,
    ):
        machine.transition(state)

    assert machine.state is RunState.COMPLETED
    assert [transition.current for transition in machine.history] == [
        RunState.CREATED,
        RunState.PREPARING,
        RunState.RUNNING,
        RunState.EVALUATING,
        RunState.FINALIZING,
        RunState.COMPLETED,
    ]


def test_run_state_machine_rejects_skips_and_terminal_transitions() -> None:
    machine = RunStateMachine()

    with pytest.raises(InvalidRunStateTransition, match="CREATED -> RUNNING"):
        machine.transition(RunState.RUNNING)

    machine.transition(RunState.FINALIZING)
    machine.transition(RunState.FAILED)
    with pytest.raises(InvalidRunStateTransition, match="FAILED -> COMPLETED"):
        machine.transition(RunState.COMPLETED)
