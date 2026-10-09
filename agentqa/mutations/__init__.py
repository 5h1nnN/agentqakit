"""Fault mutation operators and runtime support."""

from agentqa.mutations.context import (
    MutationTrialSession,
    get_current_mutation_session,
    reset_current_mutation_session,
    set_current_mutation_session,
)
from agentqa.mutations.engine import MutationEngine
from agentqa.mutations.matcher import match_mutation
from agentqa.mutations.operators import (
    InjectedMutationError,
    InjectedMutationTimeout,
    Invocation,
    InvocationResult,
    MutationOperator,
    OperatorOutcome,
    built_in_operators,
)
from agentqa.mutations.proxy import FaultProxy, FaultProxyError, FaultProxyOptions
from agentqa.mutations.scoring import calculate_mutation_score
from agentqa.mutations.wrappers import wrap_callable, wrap_retriever, wrap_tool

__all__ = [
    "FaultProxy",
    "FaultProxyError",
    "FaultProxyOptions",
    "InjectedMutationError",
    "InjectedMutationTimeout",
    "Invocation",
    "InvocationResult",
    "MutationEngine",
    "MutationOperator",
    "MutationTrialSession",
    "OperatorOutcome",
    "built_in_operators",
    "calculate_mutation_score",
    "get_current_mutation_session",
    "match_mutation",
    "reset_current_mutation_session",
    "set_current_mutation_session",
    "wrap_callable",
    "wrap_retriever",
    "wrap_tool",
]
