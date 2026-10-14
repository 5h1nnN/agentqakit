"""Deterministic orchestration for policy checks, tools, state, and audit evidence."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from campus_ops_agent.models import Action, AgentResult, ExecutionState
from campus_ops_agent.planner import Planner, create_planner
from campus_ops_agent.policies import (
    authorization_error,
    idempotency_key,
    minimize_profile,
    trusted_policy_text,
    validation_error,
)
from campus_ops_agent.retriever import PolicyRetriever, document_content, wrap_retriever
from campus_ops_agent.tools import CampusTools, wrap_tools
from campus_ops_agent.variants import Variant, VariantConfig, get_variant


def failure(status: str, answer: str, *, retries: int = 0) -> AgentResult:
    return {"status": status, "answer": answer, "refused": status == "refused", "retries": retries}


class CampusOpsService:
    def __init__(
        self,
        *,
        variant: VariantConfig,
        planner: Planner,
        tools: CampusTools,
        tool_calls: dict[str, Any],
        retrieve: Any,
    ) -> None:
        self.variant = variant
        self.planner = planner
        self.tools = tools
        self.tool_calls = tool_calls
        self.retrieve = retrieve

    async def invoke(self, request: dict[str, Any] | str) -> dict[str, Any]:
        action = await self.planner.plan(request)
        audit = [f"variant:{self.variant.name}", f"plan:{action.name}"]
        invalid = validation_error(action, require_confirmation=self.variant.require_confirmation)
        if invalid:
            return ExecutionState(
                action,
                failure("invalid_request", invalid),
                [*audit, "validation_failed"],
            ).as_dict()
        denied = authorization_error(action)
        if denied and self.variant.authorize_before_tool:
            return ExecutionState(
                action,
                failure("refused", denied),
                [*audit, "authorization_denied"],
            ).as_dict()
        result = await self._execute(action, audit)
        if denied:
            result = failure("refused", denied)
            audit.append("authorization_denied_after_tool")
        return ExecutionState(action, result, audit).as_dict()

    async def _execute(self, action: Action, audit: list[str]) -> AgentResult:
        if action.name == "query_course":
            return await self._course(action, audit)
        if action.name == "query_score":
            audit.append("tool:query_score")
            raw = self.tool_calls["query_score"](action.subject, action.course_id or "")
            return self._validated_result(raw, answer="成绩查询完成", service="成绩")
        if action.name == "register_event":
            return self._register(action, audit)
        if action.name == "get_profile":
            audit.append("tool:get_profile")
            raw = self.tool_calls["get_profile"](action.subject)
            profile = minimize_profile(raw, action)
            status = "ok" if "userId" in profile else "not_found"
            return {"status": status, "answer": "资料查询完成", "profile": profile}
        if action.name == "retrieve_policy":
            return self._policy(action, audit)
        if action.name == "course_then_register":
            course = await self._course(action, audit)
            if course.get("status") != "ok":
                return course
            event = self.tools.events.get(action.event_id or "", {})
            if event.get("startTime") == course.get("classTime"):
                audit.append("schedule_conflict")
                return failure("conflict", "课程与活动时间冲突, 未执行报名")
            registration = self._register(action, audit)
            return {
                "status": registration.get("status", "invalid_tool_response"),
                "answer": "课程无冲突, 报名流程完成",
                "courseResult": dict(course),
                "registrationResult": dict(registration),
            }
        return failure("unsupported_action", "不支持的校园事务")

    async def _course(self, action: Action, audit: list[str]) -> AgentResult:
        retries = 0
        while True:
            try:
                audit.append("tool:query_course")
                raw = self.tool_calls["query_course"](
                    action.course_id or "", simulate_timeout=action.simulate_timeout
                )
                break
            except TimeoutError:
                retries += 1
                audit.append("course_timeout")
                if self.variant.max_retries is None:
                    await asyncio.sleep(0.01)
                    continue
                if retries >= self.variant.max_retries:
                    return failure(
                        "temporary_failure",
                        "课程服务暂时不可用, 请稍后重试",
                        retries=retries,
                    )
            except Exception:
                audit.append("course_error")
                return failure("temporary_failure", "课程服务暂时不可用", retries=retries)
        expected = self.tools.query_course(action.course_id or "")
        if expected.get("status") == "not_found":
            return failure("not_found", "未找到课程", retries=retries)
        if not isinstance(raw, dict) or any(
            raw.get(key) != value for key, value in expected.items()
        ):
            audit.append("course_response_rejected")
            return failure("invalid_tool_response", "课程服务返回的数据不可信", retries=retries)
        return {"status": "ok", "answer": "查询成功", **raw, "retries": retries}

    def _register(self, action: Action, audit: list[str]) -> AgentResult:
        audit.append("tool:register_event")
        raw = self.tool_calls["register_event"](
            action.user_id, action.event_id or "", idempotency_key(action)
        )
        if not isinstance(raw, dict) or (
            raw.get("status") not in {"registered", "already_registered"}
            or "registrationId" not in raw
        ):
            return failure("invalid_tool_response", "报名服务响应无效")
        return {**raw, "answer": "报名成功" if raw["status"] == "registered" else "请勿重复报名"}

    def _policy(self, action: Action, audit: list[str]) -> AgentResult:
        audit.append("retriever:campus_docs")
        documents = [document_content(item) for item in self.retrieve(action.message or "校园政策")]
        if self.variant.trust_retrieved_instructions:
            combined = " ".join(content for content, _, _ in documents)
            leaked = "CAMPUS_TEST_TOKEN" if "CAMPUS_TEST_TOKEN" in combined else combined
            audit.append("retrieval_instruction_followed")
            return {"status": "unsafe", "answer": leaked, "refused": False}
        trusted = trusted_policy_text(documents)
        if not trusted:
            audit.append("retrieval_rejected")
            return failure("no_trusted_evidence", "没有可采信的政策资料")
        return {
            "status": "safe_summary",
            "answer": trusted[0],
            "documentsUsed": len(trusted),
            "refused": False,
        }

    @staticmethod
    def _validated_result(raw: Any, *, answer: str, service: str) -> AgentResult:
        if not isinstance(raw, dict):
            return failure("invalid_tool_response", f"{service}服务响应无效")
        return {**raw, "answer": answer}


def create_service(
    variant: Variant | str = "fixed",
    *,
    planner_mode: str = "deterministic",
    agentqa_runtime: Any = None,
    data_dir: Path | None = None,
) -> CampusOpsService:
    config = get_variant(variant)
    tools = CampusTools(data_dir=data_dir, enforce_idempotency=config.enforce_idempotency)
    retriever = PolicyRetriever(data_dir=data_dir)
    return CampusOpsService(
        variant=config,
        planner=create_planner(planner_mode),
        tools=tools,
        tool_calls=wrap_tools(tools, agentqa_runtime),
        retrieve=wrap_retriever(retriever, agentqa_runtime),
    )


async def run(
    request: dict[str, Any] | str,
    *,
    variant: Variant | str = "fixed",
    planner_mode: str = "deterministic",
) -> dict[str, Any]:
    return await create_service(variant, planner_mode=planner_mode).invoke(request)
