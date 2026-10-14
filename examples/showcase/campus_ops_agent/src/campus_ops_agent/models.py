"""Small domain models shared by the planner and deterministic executor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias, TypedDict

ActionName: TypeAlias = Literal[
    "query_course",
    "query_score",
    "register_event",
    "get_profile",
    "retrieve_policy",
    "course_then_register",
    "unsupported",
]
Role: TypeAlias = Literal["student", "staff", "guest"]


@dataclass(frozen=True, slots=True)
class Action:
    """Strict planner output consumed by policy and execution code."""

    name: ActionName
    user_id: str = "student-001"
    role: Role = "student"
    subject_user_id: str | None = None
    course_id: str | None = None
    event_id: str | None = None
    confirmation: bool = False
    simulate_timeout: bool = False
    idempotency_key: str | None = None
    message: str = ""

    @property
    def subject(self) -> str:
        return self.subject_user_id or self.user_id


class AgentResult(TypedDict, total=False):
    status: str
    answer: str
    refused: bool
    retries: int
    courseId: str
    course: str
    classTime: str
    room: str
    userId: str
    eventId: str
    registrationId: str
    score: int
    documentsUsed: int
    profile: dict[str, Any]
    courseResult: dict[str, Any]
    registrationResult: dict[str, Any]


@dataclass(slots=True)
class ExecutionState:
    """Observable state returned by both the standalone and LangGraph entrypoints."""

    action: Action
    result: AgentResult
    audit: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"result": dict(self.result), "audit": list(self.audit), "plan": asdict(self.action)}


def request_dict(value: Mapping[str, Any] | str) -> dict[str, Any]:
    """Normalize the two supported public input shapes."""

    if isinstance(value, str):
        return {"message": value}
    return dict(value)
