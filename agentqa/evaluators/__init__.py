"""Built-in deterministic evaluators and optional judge integration."""

from collections.abc import Mapping

from agentqa.evaluators.base import EvaluationRequest, Evaluator
from agentqa.evaluators.budget import BudgetEvaluator
from agentqa.evaluators.judge import (
    JudgeDecision,
    JudgeProvider,
    JudgeRequest,
    LLMJudgeEvaluator,
)
from agentqa.evaluators.output import PrivacyEvaluator, SchemaEvaluator, TextEvaluator
from agentqa.evaluators.state import StateEvaluator
from agentqa.evaluators.trace import ToolCallEvaluator, TrajectoryEvaluator


def built_in_evaluators(
    judge_provider: JudgeProvider | None = None,
) -> Mapping[str, Evaluator]:
    """Create a fresh registry for one runner instance."""
    evaluators: list[Evaluator] = [
        SchemaEvaluator(),
        TextEvaluator(),
        ToolCallEvaluator(),
        TrajectoryEvaluator(),
        StateEvaluator(),
        BudgetEvaluator(),
        PrivacyEvaluator(),
        LLMJudgeEvaluator(judge_provider),
    ]
    return {evaluator.type: evaluator for evaluator in evaluators}


__all__ = [
    "BudgetEvaluator",
    "EvaluationRequest",
    "Evaluator",
    "JudgeDecision",
    "JudgeProvider",
    "JudgeRequest",
    "LLMJudgeEvaluator",
    "PrivacyEvaluator",
    "SchemaEvaluator",
    "StateEvaluator",
    "TextEvaluator",
    "ToolCallEvaluator",
    "TrajectoryEvaluator",
    "built_in_evaluators",
]
