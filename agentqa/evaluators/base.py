"""Evaluator contracts consumed by the test runner."""

from dataclasses import dataclass
from typing import Protocol

from agentqa.models import (
    AgentTestSpec,
    EvaluationResult,
    RunContext,
    TargetResult,
    TraceRecord,
)
from agentqa.models.specs import EvaluatorSpec


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """All validated evidence and expectations available to one evaluator."""

    evaluator_spec: EvaluatorSpec
    test_spec: AgentTestSpec
    target_result: TargetResult
    trace_record: TraceRecord
    context: RunContext


class Evaluator(Protocol):
    """Evaluate one request without mutating target or trace evidence."""

    type: str
    requires_trace: bool
    runs_on_target_error: bool

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...
