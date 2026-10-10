import json
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import pytest
from pydantic import JsonValue, ValidationError

from agentqa.conventions import utc_now
from agentqa.evaluators import (
    BudgetEvaluator,
    EvaluationRequest,
    JudgeRequest,
    JudgeResponse,
    LLMJudgeEvaluator,
    PrivacyEvaluator,
    SchemaEvaluator,
    StateEvaluator,
    TextEvaluator,
    ToolCallEvaluator,
    TrajectoryEvaluator,
    built_in_evaluators,
)
from agentqa.models import (
    AgentTestSpec,
    ErrorInfo,
    RunContext,
    TargetResult,
    TraceEvent,
    TraceRecord,
)


def _spec(
    expectation: dict[str, Any],
    evaluator: dict[str, Any],
    *,
    scenario: dict[str, Any] | None = None,
    labels: dict[str, str] | None = None,
) -> AgentTestSpec:
    return AgentTestSpec.model_validate(
        {
            "metadata": {"name": "evaluation-case", "labels": labels or {}},
            "scenario": scenario or {"input": {"message": "hello"}},
            "expect": expectation,
            "evaluators": [evaluator],
        }
    )


def _event(
    span_id: str,
    *,
    kind: str,
    name: str,
    offset: int,
    parent_span_id: str = "root",
    input_value: JsonValue = None,
    attributes: dict[str, JsonValue] | None = None,
    state_diff: dict[str, JsonValue] | None = None,
    status: str = "OK",
) -> TraceEvent:
    now = utc_now()
    error = (
        ErrorInfo(code="SYNTHETIC_ERROR", message="synthetic failure")
        if status == "ERROR"
        else None
    )
    return TraceEvent.model_validate(
        {
            "trace_id": "trace-1",
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "kind": kind,
            "name": name,
            "input": input_value,
            "started_at": now + timedelta(milliseconds=offset),
            "ended_at": now + timedelta(milliseconds=offset + 1),
            "status": status,
            "error": None if error is None else error.model_dump(mode="json"),
            "attributes": attributes or {},
            "state_diff": state_diff,
        }
    )


def _trace(*events: TraceEvent, status: str = "complete") -> TraceRecord:
    if status != "complete":
        return TraceRecord.model_validate(
            {
                "trace_id": "trace-1",
                "status": status,
                "events": [event.model_dump(mode="json") for event in events],
            }
        )
    now = utc_now()
    root = TraceEvent(
        trace_id="trace-1",
        span_id="root",
        kind="AGENT",
        name="agentqa.trial",
        started_at=now,
        ended_at=now + timedelta(seconds=1),
        status="OK",
        attributes={"agentqa.runner.root": True},
    )
    return TraceRecord(
        trace_id="trace-1",
        status="complete",
        root_span_id="root",
        events=[root, *events],
    )


def _request(
    spec: AgentTestSpec,
    *,
    output: JsonValue = None,
    metadata: dict[str, JsonValue] | None = None,
    trace: TraceRecord | None = None,
    target_status: str = "OK",
) -> EvaluationRequest:
    now = utc_now()
    error = (
        ErrorInfo(code="TARGET_RUNTIME_ERROR", message="target failed")
        if target_status != "OK"
        else None
    )
    target = TargetResult.model_validate(
        {
            "status": target_status,
            "output": output,
            "error": None if error is None else error.model_dump(mode="json"),
            "started_at": now,
            "ended_at": now + timedelta(milliseconds=10),
            "metadata": metadata or {},
        }
    )
    context = RunContext(
        run_id="run-1",
        spec_hash="sha256:spec",
        started_at=now,
        random_seed=7,
        test_case_id=spec.metadata.name,
        trial_id="trial-1",
        trace_id="trace-1",
        trial_seed=11,
    )
    return EvaluationRequest(
        evaluator_spec=spec.evaluators[0],
        test_spec=spec,
        target_result=target,
        trace_record=trace or _trace(status="missing"),
        context=context,
    )


async def test_schema_evaluator_checks_structure_fields_status_and_target_failure() -> None:
    spec = _spec(
        {
            "output": {
                "requireNonEmpty": True,
                "exactFields": {"$.answer": "yes"},
                "jsonSchema": {
                    "type": "object",
                    "required": ["answer"],
                    "properties": {"answer": {"type": "string"}},
                },
                "allowedHttpStatuses": [200],
            }
        },
        {"type": "schema"},
    )

    passed = await SchemaEvaluator().evaluate(
        _request(spec, output={"answer": "yes"}, metadata={"status_code": 200})
    )
    failed = await SchemaEvaluator().evaluate(
        _request(spec, output={}, metadata={"status_code": 500})
    )
    target_failed = await SchemaEvaluator().evaluate(_request(spec, target_status="ERROR"))

    assert passed.status == "PASS"
    assert failed.status == "FAIL"
    assert {violation.type for violation in failed.violations} >= {
        "output_empty",
        "missing_output_field",
        "json_schema",
        "http_status",
    }
    assert target_failed.status == "FAIL"
    assert target_failed.violations[0].type == "target_status"


async def test_text_and_privacy_evaluators_apply_bounded_declared_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_spec = _spec(
        {
            "output": {
                "requiredPatterns": ["cannot"],
                "forbiddenPatterns": ["secret"],
                "mustRefuse": True,
            }
        },
        {"type": "text", "config": {"maxTextChars": 100}},
    )
    passed = await TextEvaluator().evaluate(
        _request(text_spec, output="I cannot provide that information.")
    )
    failed = await TextEvaluator().evaluate(_request(text_spec, output="Here is the secret."))

    privacy_spec = _spec(
        {"output": {"mustRefuse": True}},
        {"type": "privacy", "config": {"forbiddenPatterns": [r"\btoken\b"]}},
    )
    privacy = await PrivacyEvaluator().evaluate(
        _request(privacy_spec, output={"refused": False, "message": "token"})
    )

    assert passed.status == "PASS"
    assert failed.status == "FAIL"
    assert {violation.type for violation in failed.violations} == {
        "required_pattern",
        "forbidden_pattern",
        "refusal_intent",
    }
    assert privacy.status == "FAIL"
    assert {violation.type for violation in privacy.violations} == {
        "privacy_forbidden_pattern",
        "privacy_refusal",
    }

    def time_out(*args: object, **kwargs: object) -> None:
        raise TimeoutError

    monkeypatch.setattr("agentqa.evaluators.output.regex.search", time_out)
    with pytest.raises(ValueError, match="timed out"):
        await TextEvaluator().evaluate(
            _request(text_spec, output="I cannot provide that information.")
        )


async def test_tool_call_evaluator_checks_counts_schema_order_and_span_ids() -> None:
    spec = _spec(
        {
            "tools": {
                "required": ["lookup", "respond"],
                "forbidden": ["delete"],
                "minCalls": {"lookup": 1},
                "maxCalls": 2,
                "argumentSchemas": {
                    "lookup": {
                        "type": "object",
                        "required": ["course"],
                        "properties": {"course": {"type": "string"}},
                    }
                },
                "order": [{"before": "lookup", "after": "respond"}],
            }
        },
        {"type": "tool_call"},
    )
    lookup = _event(
        "lookup-span",
        kind="TOOL",
        name="lookup",
        offset=1,
        input_value={"course": "AI101"},
    )
    respond = _event("respond-span", kind="TOOL", name="respond", offset=2)
    passed = await ToolCallEvaluator().evaluate(
        _request(spec, output="ok", trace=_trace(lookup, respond))
    )
    early_respond = _event("respond-span", kind="TOOL", name="respond", offset=1)
    late_lookup = _event(
        "lookup-span",
        kind="TOOL",
        name="lookup",
        offset=2,
        input_value={"course": "AI101"},
    )
    reversed_result = await ToolCallEvaluator().evaluate(
        _request(spec, output="ok", trace=_trace(early_respond, late_lookup))
    )

    assert passed.status == "PASS"
    assert reversed_result.status == "FAIL"
    order = next(
        violation for violation in reversed_result.violations if violation.type == "tool_order"
    )
    assert order.span_id == "respond-span"


async def test_tool_argument_summary_is_not_observable() -> None:
    spec = _spec(
        {"tools": {"argumentSchemas": {"lookup": {"type": "object", "required": ["course"]}}}},
        {"type": "tool_call"},
    )
    summary: dict[str, JsonValue] = {
        "type": "dict",
        "length_bytes": 40,
        "sha256": "sha256:value",
    }
    call = _event(
        "lookup-span",
        kind="TOOL",
        name="lookup",
        offset=1,
        input_value=summary,
    )

    result = await ToolCallEvaluator().evaluate(_request(spec, output="ok", trace=_trace(call)))

    assert result.status == "NOT_OBSERVABLE"


async def test_trajectory_evaluator_checks_errors_and_required_attributes() -> None:
    spec = _spec(
        {
            "trajectory": {
                "maxErrorSpans": 0,
                "requiredSpanAttributes": [
                    {
                        "kind": "LLM",
                        "name": "chat",
                        "attribute": "gen_ai.request.model",
                        "equals": "model-a",
                    }
                ],
            }
        },
        {"type": "trajectory"},
    )
    llm = _event(
        "llm-span",
        kind="LLM",
        name="chat",
        offset=1,
        attributes={"gen_ai.request.model": "model-b"},
        status="ERROR",
    )

    result = await TrajectoryEvaluator().evaluate(_request(spec, output="ok", trace=_trace(llm)))

    assert result.status == "FAIL"
    assert {violation.type for violation in result.violations} == {
        "max_error_spans",
        "span_attribute_value",
    }
    assert all(violation.span_id == "llm-span" for violation in result.violations)


async def test_budget_evaluator_uses_observed_metrics_and_preserves_unknown() -> None:
    spec = _spec(
        {
            "budgets": {
                "maxLatencyMs": 100,
                "maxTotalTokens": 10,
                "maxSteps": 2,
                "maxLlmCalls": 1,
                "maxToolCalls": 1,
                "maxCostUsd": 1.0,
                "maxRetries": 1,
            }
        },
        {"type": "budget"},
    )
    llm = _event("llm", kind="LLM", name="chat", offset=1)
    tool = _event("tool", kind="TOOL", name="lookup", offset=2)
    passed = await BudgetEvaluator().evaluate(
        _request(
            spec,
            output="ok",
            metadata={"total_tokens": 9, "cost_usd": 0.25, "retry_count": 1},
            trace=_trace(llm, tool),
        )
    )
    failed_and_unknown = await BudgetEvaluator().evaluate(
        _request(
            spec,
            output="ok",
            metadata={"total_tokens": 11, "retry_count": 2},
            trace=_trace(llm, tool),
        )
    )

    assert passed.status == "PASS"
    assert failed_and_unknown.status == "FAIL"
    assert {violation.type for violation in failed_and_unknown.violations} == {
        "max_total_tokens",
        "max_retries",
    }
    assert failed_and_unknown.metadata["cost_usd"] is None

    unknown_only_spec = _spec(
        {"budgets": {"maxCostUsd": 1.0}},
        {"type": "budget"},
    )
    unknown = await BudgetEvaluator().evaluate(_request(unknown_only_spec, output="ok"))
    assert unknown.status == "NOT_OBSERVABLE"
    assert unknown.metadata["cost_usd"] is None


async def test_state_evaluator_supports_restricted_dsl_and_ordering() -> None:
    spec = _spec(
        {
            "state": {
                "all": [
                    {"field": "phase", "op": "equals", "value": "done"},
                    {"field": "secret", "op": "notEquals", "value": True},
                    {"field": "result", "op": "exists"},
                    {"field": "grade", "op": "in", "value": ["A", "B"]},
                    {"field": "started", "op": "before", "value": "result"},
                ]
            }
        },
        {"type": "state"},
        scenario={"input": {"message": "hello"}, "initialState": {"secret": False}},
    )
    started = _event(
        "started",
        kind="NODE",
        name="start",
        offset=1,
        state_diff={"started": True, "phase": "running"},
    )
    finished = _event(
        "finished",
        kind="NODE",
        name="finish",
        offset=2,
        state_diff={"result": "ok", "phase": "done", "grade": "A"},
    )

    result = await StateEvaluator().evaluate(
        _request(spec, output="ok", trace=_trace(started, finished))
    )

    assert result.status == "PASS"
    assert result.metadata["state_diff_events"] == 2


class FakeJudgeProvider:
    name = "fake"
    model = "fake-model"
    parameters: Mapping[str, JsonValue] = {"temperature": 0.0, "api_key": "hidden"}

    def __init__(
        self,
        responses: list[str | Mapping[str, JsonValue] | JudgeResponse],
    ) -> None:
        self.responses = responses
        self.requests: list[JudgeRequest] = []

    async def judge(
        self,
        request: JudgeRequest,
    ) -> str | Mapping[str, JsonValue] | JudgeResponse:
        self.requests.append(request)
        return self.responses.pop(0)


async def test_llm_judge_is_provider_neutral_redacted_and_retries_parse_once() -> None:
    spec = _spec(
        {"output": {"requireNonEmpty": True}},
        {"type": "llm_judge", "rubric": "The answer must be useful."},
        scenario={
            "input": {"message": "hello", "api_key": "top-secret"},
            "messages": [{"role": "user", "content": "authorization=private"}],
        },
    )
    provider = FakeJudgeProvider(
        [
            "not-json",
            {
                "verdict": "NEEDS_REVIEW",
                "score": 0.5,
                "rationale": "Evidence is ambiguous.",
                "violations": [],
            },
        ]
    )

    result = await LLMJudgeEvaluator(provider).evaluate(_request(spec, output={"answer": "ok"}))

    assert result.status == "NEEDS_REVIEW"
    assert result.metadata["attempts"] == 2
    assert "prompt_hash" in result.metadata
    assert "rubric_hash" in result.metadata
    assert result.metadata["parameters"] == {"temperature": 0.0, "api_key": "***"}
    assert [request.attempt for request in provider.requests] == [1, 2]
    assert "top-secret" not in provider.requests[0].prompt
    assert "authorization=private" not in provider.requests[0].prompt
    assert "***" in provider.requests[0].prompt
    assert "prompt" not in result.metadata


async def test_llm_judge_fail_invalid_output_and_disabled_provider() -> None:
    spec = _spec(
        {"output": {"requireNonEmpty": True}},
        {"type": "llm_judge", "rubric": "Answer correctly."},
    )
    failed_provider = FakeJudgeProvider(
        [
            json.dumps(
                {
                    "verdict": "FAIL",
                    "score": 0.0,
                    "rationale": "Incorrect.",
                    "violations": ["The answer was incorrect."],
                }
            )
        ]
    )
    failed = await LLMJudgeEvaluator(failed_provider).evaluate(_request(spec, output="wrong"))
    invalid = await LLMJudgeEvaluator(FakeJudgeProvider(["bad", "still bad"])).evaluate(
        _request(spec, output="unclear")
    )
    disabled = await LLMJudgeEvaluator().evaluate(_request(spec, output="anything"))

    assert failed.status == "FAIL"
    assert failed.violations[0].type == "llm_judge"
    assert invalid.status == "ERROR"
    assert invalid.metadata["attempts"] == 2
    assert disabled.status == "SKIPPED"


async def test_llm_judge_records_redacted_provider_accounting() -> None:
    spec = _spec(
        {"output": {"requireNonEmpty": True}},
        {"type": "llm_judge", "rubric": "Answer correctly."},
    )
    provider = FakeJudgeProvider(
        [
            JudgeResponse(
                content={
                    "verdict": "PASS",
                    "score": 1.0,
                    "rationale": "Correct.",
                    "violations": [],
                },
                metadata={
                    "usage": {"total_tokens": 42},
                    "api_key": "must-not-persist",
                },
            )
        ]
    )

    result = await LLMJudgeEvaluator(provider).evaluate(_request(spec, output="answer"))

    assert result.status == "PASS"
    assert result.metadata["provider_attempts"] == [
        {"usage": {"total_tokens": 42}, "api_key": "***"}
    ]


def test_builtin_registry_and_spec_safety_validation() -> None:
    assert set(built_in_evaluators()) == {
        "schema",
        "text",
        "tool_call",
        "trajectory",
        "state",
        "budget",
        "privacy",
        "llm_judge",
    }

    with pytest.raises(ValidationError, match="high-risk tests cannot rely only"):
        _spec(
            {"output": {"requireNonEmpty": True}},
            {"type": "llm_judge", "rubric": "Judge it."},
            labels={"risk": "high"},
        )

    with pytest.raises(ValidationError, match="exists state invariant"):
        _spec(
            {"state": {"all": [{"field": "result", "op": "exists", "value": True}]}},
            {"type": "state"},
        )

    with pytest.raises(ValidationError, match="valid regular expression"):
        _spec(
            {"output": {"requiredPatterns": ["("]}},
            {"type": "text"},
        )

    with pytest.raises(ValidationError, match="valid Draft 2020-12 schema"):
        _spec(
            {"output": {"jsonSchema": {"type": "invented"}}},
            {"type": "schema"},
        )

    with pytest.raises(ValidationError, match="argumentSchemas"):
        _spec(
            {"tools": {"argumentSchemas": {"lookup": {"type": "invented"}}}},
            {"type": "tool_call"},
        )

    with pytest.raises(ValidationError, match="text evaluator requires"):
        _spec(
            {"output": {"requireNonEmpty": True}},
            {"type": "text"},
        )

    with pytest.raises(ValidationError, match="at least one constraint"):
        _spec(
            {"tools": {}},
            {"type": "tool_call"},
        )
