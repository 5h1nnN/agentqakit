"""LangGraph adapter over the standalone CampusOps service."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from campus_ops_agent.service import create_service
from campus_ops_agent.variants import Variant


class CampusGraphState(TypedDict, total=False):
    action: str
    message: str
    user_id: str
    role: str
    subject_user_id: str
    course_id: str
    event_id: str
    confirmation: bool
    simulate_timeout: bool
    idempotency_key: str
    result: dict[str, Any]
    audit: list[str]
    plan: dict[str, Any]


def build_graph(
    variant: Variant | str = "fixed",
    *,
    planner_mode: str = "deterministic",
    agentqa_runtime: Any = None,
) -> Any:
    service = create_service(
        variant,
        planner_mode=planner_mode,
        agentqa_runtime=agentqa_runtime,
    )

    async def handle_request(state: CampusGraphState) -> CampusGraphState:
        return await service.invoke(dict(state))  # type: ignore[return-value]

    builder = StateGraph(CampusGraphState)
    builder.add_node("campus_ops", handle_request)
    builder.add_edge(START, "campus_ops")
    builder.add_edge("campus_ops", END)
    return builder.compile()


def _factory(variant: Variant) -> Callable[..., Any]:
    def factory(agentqa_runtime: Any = None) -> Any:
        return build_graph(variant, agentqa_runtime=agentqa_runtime)

    return factory


build_bug_retry = _factory("bug-retry")
build_bug_confirmation = _factory("bug-confirmation")
build_bug_authorization = _factory("bug-authorization")
build_bug_injection = _factory("bug-injection")
build_bug_idempotency = _factory("bug-idempotency")
build_fixed = _factory("fixed")

# Migration aliases for the original single-file campus fixture nomenclature.
build_v0_bug1 = build_bug_retry
build_v0_bug2 = build_bug_confirmation
build_v0_bug3 = build_bug_authorization
build_v0_bug4 = build_bug_injection
build_v1_fixed = build_fixed
