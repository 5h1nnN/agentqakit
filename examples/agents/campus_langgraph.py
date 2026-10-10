"""Synthetic campus-services LangGraph with four reproducible defect variants."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

Variant = Literal["v0-bug1", "v0-bug2", "v0-bug3", "v0-bug4", "v1-fixed"]

_COURSES = {
    "AI101": {
        "courseId": "AI101",
        "course": "人工智能导论",
        "classTime": "周一 09:00",
        "room": "A101",
    },
    "CS102": {
        "courseId": "CS102",
        "course": "数据结构",
        "classTime": "周二 14:00",
        "room": "B203",
    },
    "MATH201": {
        "courseId": "MATH201",
        "course": "离散数学",
        "classTime": "周四 10:00",
        "room": "C305",
    },
}
_PROFILES = {
    "student-001": {"userId": "student-001", "displayName": "测试学生甲", "role": "student"},
    "student-002": {"userId": "student-002", "displayName": "测试学生乙", "role": "student"},
    "staff-001": {"userId": "staff-001", "displayName": "测试教务员", "role": "staff"},
}
_SCORES = {
    ("student-001", "AI101"): {"courseId": "AI101", "score": 91},
    ("student-001", "CS102"): {"courseId": "CS102", "score": 86},
    ("student-002", "AI101"): {"courseId": "AI101", "score": 88},
}
_DOCS = (
    {
        "id": "campus-policy-1",
        "trusted": True,
        "content": "活动报名必须由本人提交, 并且必须提供活动编号。",
    },
    {
        "id": "campus-policy-2",
        "trusted": True,
        "content": "学生只能查询自己的成绩; 教务人员可按授权范围查询。",
    },
)


class CampusState(TypedDict, total=False):
    action: str
    message: str
    user_id: str
    role: str
    subject_user_id: str
    course_id: str
    event_id: str
    confirmation: bool
    simulate_timeout: bool
    result: dict[str, Any]
    audit: list[str]


def _document_content(document: Any) -> tuple[str, bool]:
    if isinstance(document, dict):
        metadata = document.get("metadata")
        mutated = isinstance(metadata, dict) and "agentqa_mutation" in metadata
        return str(document.get("content") or document.get("page_content") or ""), mutated
    metadata = getattr(document, "metadata", {})
    mutated = isinstance(metadata, dict) and "agentqa_mutation" in metadata
    return str(getattr(document, "page_content", "")), mutated


def _build_graph(variant: Variant, agentqa_runtime: Any) -> Any:
    registrations: set[tuple[str, str]] = set()

    def query_course(course_id: str, simulate_timeout: bool = False) -> dict[str, Any]:
        if simulate_timeout:
            raise TimeoutError("synthetic course service timeout")
        course = _COURSES.get(course_id)
        return dict(course) if course is not None else {"status": "not_found"}

    def query_score(subject_user_id: str, course_id: str) -> dict[str, Any]:
        score = _SCORES.get((subject_user_id, course_id))
        if score is None:
            return {"status": "not_found"}
        return {"status": "ok", "userId": subject_user_id, **score}

    def register_event(user_id: str, event_id: str) -> dict[str, Any]:
        registration = (user_id, event_id)
        if registration in registrations:
            return {"status": "already_registered", "userId": user_id, "eventId": event_id}
        registrations.add(registration)
        return {
            "status": "registered",
            "registrationId": f"reg-{user_id}-{event_id}",
            "userId": user_id,
            "eventId": event_id,
        }

    def get_profile(subject_user_id: str) -> dict[str, Any]:
        profile = _PROFILES.get(subject_user_id)
        return dict(profile) if profile is not None else {"status": "not_found"}

    def campus_docs(query: str) -> list[dict[str, Any]]:
        del query
        return [dict(document) for document in _DOCS]

    query_course_tool = agentqa_runtime.wrap_tool(query_course, name="query_course")
    query_score_tool = agentqa_runtime.wrap_tool(query_score, name="query_score")
    register_event_tool = agentqa_runtime.wrap_tool(register_event, name="register_event")
    get_profile_tool = agentqa_runtime.wrap_tool(get_profile, name="get_profile")
    retriever = agentqa_runtime.wrap_retriever(campus_docs, name="campus_docs")

    def failure(status: str, answer: str, *, retries: int = 0) -> dict[str, Any]:
        return {
            "status": status,
            "answer": answer,
            "refused": status == "refused",
            "retries": retries,
        }

    async def handle(state: CampusState) -> CampusState:
        action = str(state.get("action") or "").strip()
        user_id = str(state.get("user_id") or "student-001")
        role = str(state.get("role") or "student")
        subject_user_id = str(state.get("subject_user_id") or user_id)
        audit = [f"variant:{variant}", f"action:{action}"]

        if action == "query_course":
            course_id = str(state.get("course_id") or "")
            if not course_id:
                return {"result": failure("invalid_request", "缺少课程编号"), "audit": audit}
            retries = 0
            while True:
                try:
                    raw = query_course_tool(
                        course_id=course_id,
                        simulate_timeout=bool(state.get("simulate_timeout", False)),
                    )
                    break
                except TimeoutError:
                    retries += 1
                    audit.append("course_timeout")
                    if variant == "v0-bug1":
                        await asyncio.sleep(0.01)
                        continue
                    if retries >= 2:
                        return {
                            "result": failure(
                                "temporary_failure",
                                "课程服务暂时不可用, 请稍后重试",
                                retries=retries,
                            ),
                            "audit": audit,
                        }
                except Exception:
                    return {
                        "result": failure(
                            "temporary_failure", "课程服务暂时不可用", retries=retries
                        ),
                        "audit": [*audit, "course_error"],
                    }
            expected = _COURSES.get(course_id)
            if expected is None:
                return {"result": failure("not_found", "未找到课程"), "audit": audit}
            if not isinstance(raw, dict) or any(
                raw.get(key) != value for key, value in expected.items()
            ):
                return {
                    "result": failure("invalid_tool_response", "课程服务返回的数据不可信"),
                    "audit": [*audit, "course_response_rejected"],
                }
            return {
                "result": {"status": "ok", "answer": "查询成功", **raw, "retries": retries},
                "audit": audit,
            }

        if action == "query_score":
            if variant != "v0-bug3" and role != "staff" and subject_user_id != user_id:
                return {
                    "result": failure("refused", "无权查询他人成绩"),
                    "audit": [*audit, "authorization_denied"],
                }
            course_id = str(state.get("course_id") or "AI101")
            try:
                raw = query_score_tool(subject_user_id=subject_user_id, course_id=course_id)
            except Exception:
                return {
                    "result": failure("temporary_failure", "成绩服务暂时不可用"),
                    "audit": audit,
                }
            if not isinstance(raw, dict):
                return {
                    "result": failure("invalid_tool_response", "成绩服务响应无效"),
                    "audit": audit,
                }
            return {"result": {**raw, "answer": "成绩查询完成"}, "audit": audit}

        if action == "register_event":
            event_id = str(state.get("event_id") or "")
            missing = not event_id or not user_id or state.get("confirmation") is not True
            if role == "guest":
                return {
                    "result": failure("refused", "访客不能报名活动"),
                    "audit": [*audit, "authorization_denied"],
                }
            if missing and variant != "v0-bug2":
                return {
                    "result": failure("invalid_request", "报名需要活动编号和明确确认"),
                    "audit": [*audit, "validation_failed"],
                }
            raw = register_event_tool(user_id=user_id or "unknown", event_id=event_id or "missing")
            if not isinstance(raw, dict) or "registrationId" not in raw:
                return {
                    "result": failure("invalid_tool_response", "报名服务响应无效"),
                    "audit": audit,
                }
            return {"result": {**raw, "answer": "报名成功"}, "audit": audit}

        if action == "get_profile":
            if role != "staff" and subject_user_id != user_id:
                return {
                    "result": failure("refused", "无权查询他人资料"),
                    "audit": [*audit, "authorization_denied"],
                }
            raw = get_profile_tool(subject_user_id=subject_user_id)
            status = "ok" if "userId" in raw else "not_found"
            return {"result": {**raw, "status": status, "answer": "资料查询完成"}, "audit": audit}

        if action == "retrieve_policy":
            documents = retriever(query=str(state.get("message") or "校园政策"))
            contents = [_document_content(document) for document in documents]
            if variant == "v0-bug4":
                combined = " ".join(content for content, _ in contents)
                leaked = "CAMPUS_TEST_TOKEN" if "CAMPUS_TEST_TOKEN" in combined else combined
                return {
                    "result": {"status": "unsafe", "answer": leaked, "refused": False},
                    "audit": [*audit, "retrieval_instruction_followed"],
                }
            trusted = [
                content
                for content, mutated in contents
                if content
                and not mutated
                and "instruction" not in content.lower()
                and "token" not in content.lower()
            ]
            if not trusted:
                return {
                    "result": failure("no_trusted_evidence", "没有可采信的政策资料"),
                    "audit": [*audit, "retrieval_rejected"],
                }
            return {
                "result": {
                    "status": "safe_summary",
                    "answer": trusted[0],
                    "documentsUsed": len(trusted),
                    "refused": False,
                },
                "audit": audit,
            }

        return {"result": failure("unsupported_action", "不支持的校园事务"), "audit": audit}

    builder = StateGraph(CampusState)
    builder.add_node("handle_request", handle)
    builder.add_edge(START, "handle_request")
    builder.add_edge("handle_request", END)
    return builder.compile()


def _factory(variant: Variant) -> Callable[[Any], Any]:
    def build(agentqa_runtime: Any = None) -> Any:
        if agentqa_runtime is None:
            raise ValueError("AgentQAKit runtime injection is required")
        return _build_graph(variant, agentqa_runtime)

    return build


build_v0_bug1 = _factory("v0-bug1")
build_v0_bug2 = _factory("v0-bug2")
build_v0_bug3 = _factory("v0-bug3")
build_v0_bug4 = _factory("v0-bug4")
build_v1_fixed = _factory("v1-fixed")
