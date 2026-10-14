"""Deterministic and optional DeepSeek planners with the same Action contract."""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.request
from typing import Any, Protocol

from campus_ops_agent.models import Action, ActionName, request_dict

_COURSE = re.compile(r"\b(?:AI|CS|MATH)\d{3}\b", re.IGNORECASE)
_EVENT = re.compile(r"\b(?:event|活动)[-_]?[a-z0-9-]+\b", re.IGNORECASE)
_USER = re.compile(r"\b(?:student|staff)-\d{3}\b", re.IGNORECASE)
_ACTION_NAMES = {
    "query_course",
    "query_score",
    "register_event",
    "get_profile",
    "retrieve_policy",
    "course_then_register",
    "unsupported",
}
_ACTION_FIELDS = {
    "name",
    "user_id",
    "role",
    "subject_user_id",
    "course_id",
    "event_id",
    "confirmation",
    "simulate_timeout",
    "idempotency_key",
    "message",
}


class Planner(Protocol):
    async def plan(self, request: dict[str, Any] | str) -> Action: ...


class DeterministicPlanner:
    """Stable rules for CI; explicit structured fields always win over text extraction."""

    async def plan(self, request: dict[str, Any] | str) -> Action:
        payload = request_dict(request)
        message = str(payload.get("message") or "")
        action = str(payload.get("action") or self._intent(message))
        if action not in _ACTION_NAMES - {"unsupported"}:
            action = "unsupported"
        user_id = str(payload.get("user_id") or "student-001")
        role = str(payload.get("role") or ("staff" if user_id.startswith("staff-") else "student"))
        if role not in {"student", "staff", "guest"}:
            role = "guest"
        course = _COURSE.search(message)
        event = _EVENT.search(message)
        users = _USER.findall(message)
        subject = payload.get("subject_user_id") or (users[-1] if users else None)
        confirmation = payload.get("confirmation") is True or any(
            phrase in message.lower()
            for phrase in ("确认报名", "confirm registration", "yes, register")
        )
        course_id = payload.get("course_id") or (course.group(0).upper() if course else "")
        event_id = payload.get("event_id") or (event.group(0).lower() if event else "")
        request_key = payload.get("idempotency_key")
        return Action(
            name=action,  # type: ignore[arg-type]
            user_id=user_id,
            role=role,  # type: ignore[arg-type]
            subject_user_id=str(subject) if subject else None,
            course_id=str(course_id) or None,
            event_id=str(event_id) or None,
            confirmation=confirmation,
            simulate_timeout=payload.get("simulate_timeout") is True,
            idempotency_key=str(request_key) if request_key else None,
            message=message,
        )

    @staticmethod
    def _intent(message: str) -> ActionName:
        text = message.lower()
        if ("课程" in text or "course" in text) and ("报名" in text or "register" in text):
            return "course_then_register"
        if "成绩" in text or "score" in text or "grade" in text:
            return "query_score"
        if "资料" in text or "profile" in text:
            return "get_profile"
        if "政策" in text or "规定" in text or "policy" in text:
            return "retrieve_policy"
        if "报名" in text or "register" in text:
            return "register_event"
        if "课程" in text or "course" in text:
            return "query_course"
        return "unsupported"


class DeepSeekPlanner:
    """Optional natural-language planner. It cannot bypass deterministic policy checks."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("CAMPUS_AGENT_LLM_API_KEY", "")
        self.base_url = os.environ.get(
            "CAMPUS_AGENT_LLM_BASE_URL", "https://api.deepseek.com/chat/completions"
        )
        self.model = os.environ.get("CAMPUS_AGENT_LLM_MODEL", "deepseek-chat")
        if not self.api_key:
            raise RuntimeError("CAMPUS_AGENT_LLM_API_KEY is required for the DeepSeek planner")

    async def plan(self, request: dict[str, Any] | str) -> Action:
        payload = request_dict(request)
        baseline = await DeterministicPlanner().plan(payload)
        result = await asyncio.to_thread(self._complete, payload)
        if not isinstance(result, dict) or set(result) - _ACTION_FIELDS:
            raise ValueError("DeepSeek planner returned fields outside the Action contract")
        result.update(
            {
                "user_id": baseline.user_id,
                "role": baseline.role,
                "confirmation": baseline.confirmation,
                "simulate_timeout": baseline.simulate_timeout,
                "idempotency_key": baseline.idempotency_key,
                "message": baseline.message,
            }
        )
        return _validated_action(result)

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        schema = {
            "name": "one of query_course, query_score, register_event, get_profile, "
            "retrieve_policy, course_then_register, unsupported",
            "user_id": "string",
            "role": "student, staff, or guest",
            "subject_user_id": "string or null",
            "course_id": "string or null",
            "event_id": "string or null",
            "confirmation": "boolean",
            "idempotency_key": "string or null",
        }
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "Convert the campus request to JSON only. "
                        "Do not decide authorization. "
                        f"Contract: {json.dumps(schema)}",
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            }
        ).encode()
        http_request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=20) as response:
            decoded = json.loads(response.read(1_000_000))
        return json.loads(decoded["choices"][0]["message"]["content"])


def create_planner(mode: str = "deterministic") -> Planner:
    if mode == "deterministic":
        return DeterministicPlanner()
    if mode == "deepseek":
        return DeepSeekPlanner()
    raise ValueError("planner mode must be 'deterministic' or 'deepseek'")


def _validated_action(value: dict[str, Any]) -> Action:
    """Fail closed when model output does not match the runtime Action contract."""

    if set(value) - _ACTION_FIELDS:
        raise ValueError("DeepSeek planner returned fields outside the Action contract")
    name = value.get("name")
    role = value.get("role", "student")
    if name not in _ACTION_NAMES:
        raise ValueError("DeepSeek planner returned an unsupported action name")
    if role not in {"student", "staff", "guest"}:
        raise ValueError("DeepSeek planner returned an unsupported role")
    for field_name in ("confirmation", "simulate_timeout"):
        if field_name in value and not isinstance(value[field_name], bool):
            raise ValueError(f"DeepSeek planner field {field_name} must be boolean")
    for field_name in (
        "user_id",
        "subject_user_id",
        "course_id",
        "event_id",
        "idempotency_key",
        "message",
    ):
        if (
            field_name in value
            and value[field_name] is not None
            and not isinstance(value[field_name], str)
        ):
            raise ValueError(f"DeepSeek planner field {field_name} must be a string or null")
    return Action(**value)  # type: ignore[arg-type]
