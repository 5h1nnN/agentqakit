"""Restricted non-executable state invariant evaluator."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from agentqa.evaluators.base import EvaluationRequest
from agentqa.evaluators.utils import TraceView, evaluator_result, path_value
from agentqa.models import EvaluationResult, TraceEvent, Violation


def _merge_state(current: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        existing = current.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _merge_state(existing, value)
        else:
            current[key] = deepcopy(value)


class StateEvaluator:
    """Evaluate equals/notEquals/exists/in/before over observed state diffs."""

    type = "state"
    requires_trace = True
    runs_on_target_error = True

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        expectation = request.test_spec.expect.state
        if expectation is None:
            raise ValueError("state evaluator requires expect.state")
        view = TraceView(request.trace_record)
        state = deepcopy(request.test_spec.scenario.initial_state)
        observations: dict[str, tuple[int, TraceEvent | None]] = {}
        paths = {invariant.field for invariant in expectation.all}
        paths.update(
            str(invariant.value) for invariant in expectation.all if invariant.op == "before"
        )
        for path in paths:
            _, found = path_value(state, path)
            if found:
                observations[path] = (-1, None)

        state_events = [event for event in view.events if event.state_diff is not None]
        for index, event in enumerate(state_events):
            assert event.state_diff is not None
            for path in paths:
                _, found = path_value(event.state_diff, path)
                if found:
                    observations.setdefault(path, (index, event))
            _merge_state(state, event.state_diff)

        violations: list[Violation] = []
        unobservable: list[str] = []
        for invariant in expectation.all:
            actual, found = path_value(state, invariant.field)
            observation = observations.get(invariant.field)
            span_id = (
                None if observation is None or observation[1] is None else observation[1].span_id
            )
            if invariant.op == "exists":
                if not found:
                    violations.append(
                        Violation(
                            type="state_exists",
                            message=f"state field {invariant.field!r} did not exist",
                        )
                    )
            elif invariant.op == "before":
                other_path = str(invariant.value)
                other = observations.get(other_path)
                if observation is None or other is None:
                    unobservable.append(
                        f"state update order for {invariant.field!r} and {other_path!r} "
                        "was not observed"
                    )
                elif observation[0] >= other[0]:
                    violations.append(
                        Violation(
                            type="state_before",
                            message=(
                                f"state field {invariant.field!r} was not observed before "
                                f"{other_path!r}"
                            ),
                            span_id=(None if other[1] is None else other[1].span_id),
                        )
                    )
            elif not found:
                unobservable.append(f"state field {invariant.field!r} was not observed")
            elif invariant.op == "equals" and actual != invariant.value:
                violations.append(
                    Violation(
                        type="state_equals",
                        message=f"state field {invariant.field!r} did not equal the expected value",
                        span_id=span_id,
                    )
                )
            elif invariant.op == "notEquals" and actual == invariant.value:
                violations.append(
                    Violation(
                        type="state_not_equals",
                        message=f"state field {invariant.field!r} equaled a forbidden value",
                        span_id=span_id,
                    )
                )
            elif invariant.op == "in":
                allowed = invariant.value
                if not isinstance(allowed, list):
                    raise ValueError("in invariant requires a list")
                if actual not in allowed:
                    violations.append(
                        Violation(
                            type="state_in",
                            message=f"state field {invariant.field!r} was outside the allowed set",
                            span_id=span_id,
                        )
                    )

        return evaluator_result(
            self.type,
            violations=violations,
            unobservable=list(dict.fromkeys(unobservable)),
            metadata={"state_diff_events": len(state_events)},
        )
