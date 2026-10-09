"""Evaluator contract consumed by the test runner."""

from collections.abc import Sequence
from typing import Protocol

from agentqa.models import EvaluationResult, RunContext, TargetResult, TraceEvent
from agentqa.models.specs import EvaluatorSpec


class Evaluator(Protocol):
    """Evaluate a target result and optional trace against one evaluator spec."""

    type: str

    async def evaluate(
        self,
        spec: EvaluatorSpec,
        target_result: TargetResult,
        trace: Sequence[TraceEvent] | None,
        context: RunContext,
    ) -> EvaluationResult: ...
