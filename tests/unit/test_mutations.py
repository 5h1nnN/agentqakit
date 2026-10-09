from datetime import UTC, datetime
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from agentqa.models import (
    AgentTestSpec,
    ErrorInfo,
    EvaluationResult,
    InterceptionPoint,
    MutationApplication,
    MutationMatch,
    MutationScoreReport,
    MutationSpec,
    RunContext,
    TargetResult,
)
from agentqa.models import TestResult as AgentTestResult
from agentqa.mutations import (
    InjectedMutationError,
    InjectedMutationTimeout,
    MutationEngine,
    MutationTrialSession,
    calculate_mutation_score,
    match_mutation,
    reset_current_mutation_session,
    set_current_mutation_session,
    wrap_retriever,
    wrap_tool,
)
from agentqa.tracing import InMemoryTraceStore
from agentqa.tracing.context import reset_current_trace_store, set_current_trace_store


def _mutation(
    mutation_id: str,
    operation: str,
    *,
    kind: str = "tool",
    name: str = "campus.lookup",
    when: dict[str, Any] | None = None,
    **operation_values: Any,
) -> MutationSpec:
    return MutationSpec.model_validate(
        {
            "metadata": {"name": mutation_id},
            "target": {"kind": kind, "name": name},
            "when": when or {},
            "operation": {"type": operation, **operation_values},
        }
    )


def _session(
    mutation_id: str,
    *,
    seed: int = 42,
    labels: dict[str, str] | None = None,
) -> MutationTrialSession:
    return MutationTrialSession(
        RunContext(
            run_id="run-1",
            spec_hash="spec-1",
            started_at=datetime.now(UTC),
            random_seed=seed,
            test_case_id="case-1",
            trial_id="trial-1",
            trace_id="trace-1",
            mutation_id=mutation_id,
            trial_seed=seed,
            test_labels=labels or {},
        )
    )


def _activate(
    spec: MutationSpec,
) -> tuple[MutationEngine, MutationTrialSession, InMemoryTraceStore, Any, Any]:
    session = _session(spec.metadata.name)
    store = InMemoryTraceStore()
    mutation_token = set_current_mutation_session(session)
    trace_token = set_current_trace_store(store)
    return MutationEngine({spec.metadata.name: spec}), session, store, mutation_token, trace_token


def _deactivate(mutation_token: Any, trace_token: Any) -> None:
    reset_current_trace_store(trace_token)
    reset_current_mutation_session(mutation_token)


def test_matcher_combines_conditions_and_probability_deterministically() -> None:
    spec = _mutation(
        "match-all",
        "wrong_value",
        when={
            "callIndex": 1,
            "arguments": {"payload.city": "Hangzhou"},
            "argumentExists": ["payload.studentId"],
            "testLabels": {"suite": "campus"},
            "probability": 0.75,
            "mode": "every",
        },
        field="status",
        value="closed",
    )
    point = InterceptionPoint(kind="tool", name="campus.lookup")
    arguments = {"payload": {"city": "Hangzhou", "studentId": "S-1"}}
    first = match_mutation(
        spec,
        point,
        arguments,
        1,
        _session("match-all", labels={"suite": "campus"}),
    )
    second = match_mutation(
        spec,
        point,
        arguments,
        1,
        _session("match-all", labels={"suite": "campus"}),
    )

    assert first == second
    assert first.probability_sample == second.probability_sample
    assert first.conditions["when.arguments.payload.city"] is True
    assert first.conditions["when.argumentExists.payload.studentId"] is True

    missed = match_mutation(
        spec,
        point,
        {"payload": {"city": "Shanghai"}},
        2,
        _session("match-all", labels={"suite": "other"}),
    )
    assert missed.matched is False
    assert missed.reason == "condition did not match: when.callIndex"


@pytest.mark.parametrize(
    "document",
    [
        {
            "target": {"kind": "tool", "name": "campus.lookup"},
            "operation": {"type": "tool_timeout"},
        },
        {
            "target": {"kind": "tool", "name": "campus.lookup"},
            "operation": {"type": "missing_field"},
        },
        {
            "target": {"kind": "tool", "name": "campus.lookup"},
            "operation": {"type": "wrong_value", "field": "answer"},
        },
        {
            "target": {"kind": "retriever", "name": "campus.docs"},
            "operation": {"type": "tool_exception"},
        },
    ],
)
def test_mutation_operation_schema_rejects_invalid_parameters(
    document: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        MutationSpec.model_validate(
            {
                "metadata": {"name": "invalid"},
                **document,
            }
        )


def test_tool_timeout_and_exception_raise_safe_injected_errors() -> None:
    timeout = _mutation("timeout", "tool_timeout", delayMs=1)
    engine, session, store, mutation_token, trace_token = _activate(timeout)
    wrapped = wrap_tool(lambda value: value, engine, name="campus.lookup")
    try:
        with pytest.raises(InjectedMutationTimeout, match="injected tool timeout"):
            wrapped("secret-input")
    finally:
        _deactivate(mutation_token, trace_token)

    assert session.applications[0].triggered is True
    assert session.applications[0].span_id
    assert store.get_events("trace-1")[0].status == "ERROR"

    exception = _mutation("exception", "tool_exception")
    engine, session, _, mutation_token, trace_token = _activate(exception)
    wrapped = wrap_tool(lambda: "unused", engine, name="campus.lookup")
    try:
        with pytest.raises(InjectedMutationError, match="injected tool exception"):
            wrapped()
    finally:
        _deactivate(mutation_token, trace_token)
    assert session.applications[0].triggered is True


async def test_async_tool_callable_uses_the_same_mutation_engine() -> None:
    spec = _mutation("async-invalid", "invalid_json")
    engine, session, _, mutation_token, trace_token = _activate(spec)
    calls = 0

    async def lookup(query: str) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"answer": query}

    wrapped = wrap_tool(lookup, engine, name="campus.lookup")
    try:
        assert await wrapped("calendar") == "{agentqa-invalid-json"
    finally:
        _deactivate(mutation_token, trace_token)

    assert calls == 1
    assert session.applications[0].triggered is True


@pytest.mark.parametrize(
    ("operation", "values", "expected"),
    [
        ("invalid_json", {}, "{agentqa-invalid-json"),
        ("missing_field", {"field": "profile.email"}, {"profile": {"name": "Ada"}}),
        (
            "wrong_value",
            {"field": "profile.name", "value": "Grace"},
            {"profile": {"name": "Grace", "email": "ada@example.test"}},
        ),
    ],
)
def test_tool_result_mutations_execute_real_callable_and_record_trace(
    operation: str,
    values: dict[str, Any],
    expected: Any,
) -> None:
    spec = _mutation(operation, operation, **values)
    engine, session, store, mutation_token, trace_token = _activate(spec)
    calls = 0

    def lookup() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"profile": {"name": "Ada", "email": "ada@example.test"}}

    wrapped = wrap_tool(lookup, engine, name="campus.lookup")
    try:
        assert wrapped() == expected
    finally:
        _deactivate(mutation_token, trace_token)

    assert calls == 1
    application = session.applications[0]
    assert application.triggered is True
    event = store.get_events("trace-1")[0]
    assert event.span_id == application.span_id
    assert event.attributes["agentqa.mutation.triggered"] is True


@pytest.mark.parametrize("operation", ["duplicate_call_result", "stale_result"])
def test_cached_result_mutations_need_a_real_prior_call(operation: str) -> None:
    spec = _mutation(operation, operation)
    engine, session, _, mutation_token, trace_token = _activate(spec)
    next_value = 0

    def lookup() -> int:
        nonlocal next_value
        next_value += 1
        return next_value

    wrapped = wrap_tool(lookup, engine, name="campus.lookup")
    try:
        assert wrapped() == 1
        assert wrapped() == 1
    finally:
        _deactivate(mutation_token, trace_token)

    assert [item.triggered for item in session.applications] == [False, True]
    expected_calls = 1 if operation == "duplicate_call_result" else 2
    assert next_value == expected_calls


@pytest.mark.parametrize(
    ("operation", "assertion"),
    [
        ("empty_retrieval", "empty"),
        ("irrelevant_document", "irrelevant_document"),
        ("indirect_prompt_injection", "CAMPUS_TEST_TOKEN"),
    ],
)
def test_retrieval_mutations_replace_real_retriever_results(
    operation: str,
    assertion: str,
) -> None:
    spec = _mutation(
        operation,
        operation,
        kind="retriever",
        name="campus.docs",
    )
    engine, session, _, mutation_token, trace_token = _activate(spec)
    calls = 0

    def retrieve(query: str) -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        return [{"page_content": f"real result for {query}"}]

    wrapped = wrap_retriever(retrieve, engine, name="campus.docs")
    try:
        result = wrapped("calendar")
    finally:
        _deactivate(mutation_token, trace_token)

    if assertion == "empty":
        assert result == []
    else:
        assert assertion in str(result)
    assert calls == 0
    assert session.applications[0].triggered is True


def test_non_matching_mutation_passes_through_and_records_miss() -> None:
    spec = _mutation(
        "miss",
        "wrong_value",
        name="different.tool",
        field="answer",
        value="mutated",
    )
    engine, session, _, mutation_token, trace_token = _activate(spec)
    wrapped = wrap_tool(lambda: {"answer": "real"}, engine, name="campus.lookup")
    try:
        assert wrapped() == {"answer": "real"}
    finally:
        _deactivate(mutation_token, trace_token)

    assert session.applications[0].triggered is False
    assert session.applications[0].match.conditions["target.name"] is False


def test_langchain_structured_tool_keeps_its_public_interface() -> None:
    tools = pytest.importorskip("langchain_core.tools")
    structured_tool_type = tools.StructuredTool

    def lookup(course: str) -> dict[str, str]:
        """Look up one course."""
        return {"course": course, "room": "A-101"}

    original = structured_tool_type.from_function(lookup, name="campus.lookup")
    spec = _mutation(
        "structured-wrong-room",
        "wrong_value",
        field="room",
        value="B-202",
    )
    engine, session, _, mutation_token, trace_token = _activate(spec)
    wrapped = wrap_tool(original, engine)
    try:
        result = wrapped.invoke({"course": "AI"})
    finally:
        _deactivate(mutation_token, trace_token)

    assert result == {"course": "AI", "room": "B-202"}
    assert wrapped.name == original.name
    assert session.applications[0].triggered is True


def _test_spec(name: str, mutation_id: str) -> AgentTestSpec:
    return AgentTestSpec.model_validate(
        {
            "metadata": {"name": name},
            "scenario": {"input": {"message": "hello"}},
            "mutation": mutation_id,
            "expect": {"output": {"requireNonEmpty": True}},
            "evaluators": [{"type": "schema"}],
        }
    )


def _application(mutation_id: str, triggered: bool) -> MutationApplication:
    now = datetime.now(UTC)
    return MutationApplication(
        mutation_id=mutation_id,
        operation_type="tool_exception",
        interception=InterceptionPoint(kind="tool", name="campus.lookup"),
        call_index=1,
        triggered=triggered,
        match=MutationMatch(
            mutation_id=mutation_id,
            matched=triggered,
            conditions={"target.name": triggered},
            reason=None if triggered else "condition did not match: target.name",
            probability_sample=0.1,
        ),
        invocation_summary={"sha256": "abc", "bytes": 1},
        behavior={"action": "raise" if triggered else "pass_through"},
        test_case_id=f"test-{mutation_id}",
        trial_id=f"trial-{mutation_id}",
        trace_id=f"trace-{mutation_id}",
        span_id=f"span-{mutation_id}",
        applied_at=now,
    )


def _result(
    mutation_id: str,
    *,
    evaluation: Literal["PASS", "FAIL", "ERROR", "SKIPPED", "NOT_OBSERVABLE", "NEEDS_REVIEW"],
    triggered: bool,
    target_status: Literal["OK", "ERROR", "TIMEOUT"] = "OK",
) -> AgentTestResult:
    now = datetime.now(UTC)
    error = (
        None
        if target_status == "OK"
        else ErrorInfo(code="TARGET_RUNTIME_ERROR", message="target failed")
    )
    evaluations = [
        EvaluationResult(
            evaluator_type="schema",
            status=evaluation,
        )
    ]
    return AgentTestResult(
        test_case_id=f"test-{mutation_id}",
        trial_id=f"trial-{mutation_id}",
        trace_id=f"trace-{mutation_id}",
        trial_index=1,
        trial_seed=1,
        started_at=now,
        ended_at=now,
        passed=target_status == "OK" and evaluation == "PASS",
        evaluations=evaluations,
        target_result=TargetResult(
            status=target_status,
            error=error,
            started_at=now,
            ended_at=now,
        ),
        mutation_applications=[_application(mutation_id, triggered)],
    )


def test_mutation_score_reports_all_outcome_classes_without_inflating_denominator() -> None:
    mutations = {
        name: _mutation(name, "tool_exception")
        for name in ("killed-a", "killed-b", "survived", "not-reached", "invalid")
    }
    specs = [_test_spec(f"test-{name}", name) for name in mutations]
    results = [
        _result("killed-a", evaluation="FAIL", triggered=True),
        _result("killed-b", evaluation="SKIPPED", triggered=True, target_status="ERROR"),
        _result("survived", evaluation="PASS", triggered=True),
        _result("not-reached", evaluation="PASS", triggered=False),
        _result("invalid", evaluation="ERROR", triggered=True),
    ]

    report = calculate_mutation_score(mutations, specs, results)

    assert report.overall.killed == 2
    assert report.overall.survived == 1
    assert report.overall.not_reached == 1
    assert report.overall.invalid == 1
    assert report.overall.score == pytest.approx(2 / 3)
    assert {outcome.status for outcome in report.outcomes} == {
        "KILLED",
        "SURVIVED",
        "NOT_REACHED",
        "INVALID",
    }

    tampered = report.model_dump(mode="json")
    tampered["overall"] = {
        "killed": 1,
        "survived": 1,
        "not_reached": 1,
        "invalid": 1,
        "score": 0.5,
    }
    with pytest.raises(ValidationError, match="recomputable"):
        MutationScoreReport.model_validate(tampered)
