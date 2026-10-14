"""Authorization, confirmation, privacy, and retrieval trust decisions."""

from __future__ import annotations

import re
from typing import Any

from campus_ops_agent.models import Action

_INSTRUCTION = re.compile(
    r"(ignore (?:all |previous )?instructions?|system prompt|reveal|token|密钥|忽略.*指令)",
    re.IGNORECASE,
)


def authorization_error(action: Action) -> str | None:
    if action.name == "register_event" and action.role == "guest":
        return "访客不能报名活动"
    if (
        action.name in {"query_score", "get_profile"}
        and action.role != "staff"
        and action.subject != action.user_id
    ):
        return "无权查询他人的隐私信息"
    return None


def validation_error(action: Action, *, require_confirmation: bool) -> str | None:
    if (
        action.name in {"query_course", "query_score", "course_then_register"}
        and not action.course_id
    ):
        return "缺少课程编号"
    if action.name in {"register_event", "course_then_register"}:
        if not action.event_id:
            return "报名需要活动编号"
        if require_confirmation and not action.confirmation:
            return "报名需要明确确认"
    return None


def idempotency_key(action: Action) -> str:
    return action.idempotency_key or f"registration:{action.user_id}:{action.event_id}"


def minimize_profile(raw: dict[str, Any], action: Action) -> dict[str, Any]:
    if "userId" not in raw:
        return {"status": "not_found"}
    allowed = {"userId", "displayName", "role"}
    if action.subject == action.user_id:
        allowed.add("email")
    return {key: value for key, value in raw.items() if key in allowed}


def trusted_policy_text(documents: list[tuple[str, bool, bool]]) -> list[str]:
    return [
        content
        for content, mutated, declared_trusted in documents
        if content and declared_trusted and not mutated and not _INSTRUCTION.search(content)
    ]
