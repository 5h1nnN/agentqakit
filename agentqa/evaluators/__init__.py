"""Built-in deterministic evaluators and optional judge integration."""

from collections.abc import Mapping

from agentqa.evaluators.base import EvaluationRequest, Evaluator
from agentqa.evaluators.budget import BudgetEvaluator
from agentqa.evaluators.deepseek import (
    DeepSeekAPIError,
    DeepSeekConfigurationError,
    DeepSeekJudgeProvider,
    DeepSeekJudgeSettings,
    DeepSeekProvider,
    DeepSeekSettings,
    configured_deepseek_provider,
    load_judge_environment,
    load_llm_environment,
)
from agentqa.evaluators.judge import (
    JudgeDecision,
    JudgeProvider,
    JudgeProviderLike,
    JudgeRequest,
    JudgeResponse,
    LLMJudgeEvaluator,
)
from agentqa.evaluators.output import PrivacyEvaluator, SchemaEvaluator, TextEvaluator
from agentqa.evaluators.state import StateEvaluator
from agentqa.evaluators.trace import ToolCallEvaluator, TrajectoryEvaluator
from agentqa.llm import (
    LLMCallEvidence,
    StructuredLLMProvider,
    StructuredRequest,
    StructuredResponse,
)


def built_in_evaluators(
    judge_provider: JudgeProviderLike | None = None,
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
    "DeepSeekAPIError",
    "DeepSeekConfigurationError",
    "DeepSeekJudgeProvider",
    "DeepSeekJudgeSettings",
    "DeepSeekProvider",
    "DeepSeekSettings",
    "EvaluationRequest",
    "Evaluator",
    "JudgeDecision",
    "JudgeProvider",
    "JudgeProviderLike",
    "JudgeRequest",
    "JudgeResponse",
    "LLMCallEvidence",
    "LLMJudgeEvaluator",
    "PrivacyEvaluator",
    "SchemaEvaluator",
    "StateEvaluator",
    "StructuredLLMProvider",
    "StructuredRequest",
    "StructuredResponse",
    "TextEvaluator",
    "ToolCallEvaluator",
    "TrajectoryEvaluator",
    "built_in_evaluators",
    "configured_deepseek_provider",
    "load_judge_environment",
    "load_llm_environment",
]
