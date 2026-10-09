import asyncio
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from agentqa.conventions import utc_now
from agentqa.evaluators import EvaluationRequest
from agentqa.models import (
    AgentTestSpec,
    EvaluationResult,
    HttpTargetSpec,
    LangGraphTargetSpec,
    MutationSpec,
    RunContext,
    TargetResult,
    Violation,
)
from agentqa.models import (
    TestInput as AgentTestInput,
)
from agentqa.replay import CassetteFormatError
from agentqa.replay.bundle import extract_failure_bundle
from agentqa.runner import RunnerOptions, get_current_run_context
from agentqa.runner import TestRunner as AgentTestRunner
from agentqa.storage import ArtifactError
from agentqa.targets import TargetSafetyError
from agentqa.targets.langgraph import LangGraphTargetAdapter


@dataclass(frozen=True, slots=True)
class Outcome:
    passed: bool = True
    latency_ms: int = 10
    total_tokens: int = 100
    steps: int = 2
    error: Exception | None = None
    delay_seconds: float = 0.0


class FakeAdapter:
    def __init__(
        self,
        outcomes: Sequence[Outcome],
        *,
        prepare_error: Exception | None = None,
        cleanup_error: Exception | None = None,
        prepare_delay_seconds: float = 0.0,
        cleanup_delay_seconds: float = 0.0,
    ) -> None:
        self.outcomes = list(outcomes)
        self.prepare_error = prepare_error
        self.cleanup_error = cleanup_error
        self.prepare_delay_seconds = prepare_delay_seconds
        self.cleanup_delay_seconds = cleanup_delay_seconds
        self.prepare_calls = 0
        self.cleanup_calls = 0
        self.invoke_calls: list[tuple[str, str]] = []
        self.contexts: list[RunContext] = []

    async def prepare(self, context: RunContext) -> None:
        self.prepare_calls += 1
        if self.prepare_delay_seconds:
            await asyncio.sleep(self.prepare_delay_seconds)
        if self.prepare_error is not None:
            raise self.prepare_error

    async def invoke(self, test_input: AgentTestInput, context: RunContext) -> TargetResult:
        current = get_current_run_context()
        assert current == context
        assert context.trial_id is not None
        self.invoke_calls.append((test_input.test_case_id, context.trial_id))
        self.contexts.append(context)
        outcome = self.outcomes.pop(0)
        if outcome.delay_seconds:
            await asyncio.sleep(outcome.delay_seconds)
        if outcome.error is not None:
            raise outcome.error
        started_at = utc_now()
        return TargetResult(
            status="OK",
            output={"passed": outcome.passed},
            started_at=started_at,
            ended_at=started_at + timedelta(milliseconds=outcome.latency_ms),
            metadata={"total_tokens": outcome.total_tokens, "steps": outcome.steps},
        )

    async def cleanup(self, context: RunContext) -> None:
        self.cleanup_calls += 1
        if self.cleanup_delay_seconds:
            await asyncio.sleep(self.cleanup_delay_seconds)
        if self.cleanup_error is not None:
            raise self.cleanup_error


class ResultEvaluator:
    type = "schema"
    requires_trace = False
    runs_on_target_error = False

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        assert request.trace_record.status == "missing"
        output = request.target_result.output
        passed = isinstance(output, dict) and output.get("passed") is True
        violations = [] if passed else [Violation(type="output", message="output failed")]
        return EvaluationResult(
            evaluator_type=request.evaluator_spec.type,
            status="PASS" if passed else "FAIL",
            score=1.0 if passed else 0.0,
            violations=violations,
        )


class RaisingEvaluator:
    type = "schema"
    requires_trace = False
    runs_on_target_error = False

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        raise ValueError("sensitive evaluator detail")


class TraceRequiredEvaluator:
    type = "schema"
    requires_trace = True
    runs_on_target_error = False

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        raise AssertionError("trace evaluator must not run without complete evidence")


class SlowEvaluator:
    type = "schema"
    requires_trace = False
    runs_on_target_error = False

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        await asyncio.sleep(60)
        raise AssertionError("timed-out evaluator continued")


class BlockingAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__([Outcome()])
        self.started = asyncio.Event()

    async def invoke(self, test_input: AgentTestInput, context: RunContext) -> TargetResult:
        self.started.set()
        await asyncio.sleep(60)
        raise AssertionError("cancelled invocation continued")


class InputMutationAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__([])
        self.observed_values: list[int] = []

    async def invoke(self, test_input: AgentTestInput, context: RunContext) -> TargetResult:
        nested = test_input.input["nested"]
        assert isinstance(nested, dict)
        value = nested["value"]
        assert isinstance(value, int)
        self.observed_values.append(value)
        nested["value"] = 99
        started_at = utc_now()
        return TargetResult(
            status="OK",
            output={"passed": True},
            started_at=started_at,
            ended_at=started_at,
        )


def _target(*, timeout_ms: int = 1000) -> HttpTargetSpec:
    return HttpTargetSpec.model_validate(
        {
            "apiVersion": "agentqa.dev/v1alpha1",
            "kind": "Target",
            "metadata": {"name": "fake-target"},
            "type": "http",
            "url": "http://localhost:8000/chat",
            "timeoutMs": timeout_ms,
        }
    )


def _test_spec(
    name: str,
    *,
    trials: int = 1,
    strategy: str = "all",
    min_pass_rate: float | None = None,
) -> AgentTestSpec:
    trial_policy: dict[str, Any] = {"strategy": strategy}
    if min_pass_rate is not None:
        trial_policy["minPassRate"] = min_pass_rate
    return AgentTestSpec.model_validate(
        {
            "apiVersion": "agentqa.dev/v1alpha1",
            "kind": "AgentTest",
            "metadata": {"name": name},
            "scenario": {"input": {"message": name}},
            "expect": {"output": {"requireNonEmpty": True}},
            "evaluators": [{"type": "schema"}],
            "trials": trials,
            "trialPolicy": trial_policy,
        }
    )


def _options(tmp_path: Path, **overrides: Any) -> RunnerOptions:
    values: dict[str, Any] = {
        "artifact_directory": tmp_path / "artifacts",
        "random_seed": 42,
        "prepare_timeout_seconds": 1.0,
        "item_timeout_seconds": 1.0,
        "evaluator_timeout_seconds": 1.0,
        "cleanup_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return RunnerOptions(**values)


async def test_runner_continues_after_target_error_and_persists_results(tmp_path: Path) -> None:
    adapter = FakeAdapter(
        [
            Outcome(passed=True, latency_ms=100, total_tokens=10, steps=1),
            Outcome(error=ValueError("sensitive target detail")),
            Outcome(passed=True, latency_ms=300, total_tokens=30, steps=3),
        ]
    )
    spec = _test_spec("alternating", trials=3)
    runner = AgentTestRunner(adapter, {"schema": ResultEvaluator()}, options=_options(tmp_path))

    result = await runner.run([spec], _target())

    assert result.status == "COMPLETED"
    assert result.exit_code == 1
    assert [trial.passed for trial in result.results] == [True, False, True]
    assert result.results[1].target_result.error is not None
    assert result.results[1].target_result.error.code == "TARGET_RUNTIME_ERROR"
    assert "sensitive target detail" not in result.results[1].target_result.error.message
    assert adapter.prepare_calls == 1
    assert adapter.cleanup_calls == 1
    assert len(adapter.invoke_calls) == 3
    case = result.summary.cases[0]
    assert case.passed is False
    assert case.inconsistent is True
    assert case.pass_rate == pytest.approx(2 / 3)
    assert case.latency_ms.count == 3
    assert case.total_tokens.count == 2
    assert case.total_tokens.mean == pytest.approx(20)
    assert get_current_run_context() is None

    root = tmp_path / "artifacts" / "runs" / result.run_id
    run_record = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert run_record["status"] == "COMPLETED"
    assert [item["current"] for item in run_record["history"]] == [
        "CREATED",
        "PREPARING",
        "RUNNING",
        "EVALUATING",
        "FINALIZING",
        "COMPLETED",
    ]
    assert (root / "summary.json").is_file()
    summary_record = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary_record["exit_code"] == 1
    assert summary_record["summary"]["inconsistent_cases"] == 1
    assert len((root / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 3


async def test_failed_trial_writes_a_verified_redacted_reproduction_bundle(
    tmp_path: Path,
) -> None:
    runner = AgentTestRunner(
        FakeAdapter([Outcome(passed=False)]),
        {"schema": ResultEvaluator()},
        options=_options(tmp_path, original_command="agentqa test run test.yaml token=private"),
    )

    result = await runner.run([_test_spec("bundle-failure")], _target())

    trial = result.results[0]
    assert trial.failure_bundle_ref is not None
    bundle = tmp_path / "artifacts" / "runs" / result.run_id / trial.failure_bundle_ref
    assert bundle.is_file()
    assert bundle.with_name(f"{bundle.name}.sha256").is_file()

    extracted = tmp_path / "extracted"
    manifest = extract_failure_bundle(bundle, extracted)
    assert manifest["trial_seed"] == trial.trial_seed
    assert {
        "manifest.yaml",
        "target.yaml",
        "test.yaml",
        "result.json",
        "trace.json",
        "cassette.jsonl",
        "environment.json",
        "README.md",
        "checksums.txt",
    }.issubset({path.name for path in extracted.iterdir()})
    readme = (extracted / "README.md").read_text(encoding="utf-8")
    assert "private" not in readme
    assert "agentqa replay" in readme

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            payload = b"{}\n" if info.filename == "result.json" else source.read(info)
            target.writestr(info, payload)
    with pytest.raises(CassetteFormatError, match="checksum mismatch"):
        extract_failure_bundle(tampered, tmp_path / "tampered-output")


async def test_trial_policies_and_metric_percentiles(tmp_path: Path) -> None:
    outcomes = [
        Outcome(True, 100, 10, 1),
        Outcome(False, 200, 20, 2),
        Outcome(True, 300, 30, 3),
    ] * 3
    specs = [
        _test_spec("all-policy", trials=3, strategy="all"),
        _test_spec("majority-policy", trials=3, strategy="majority"),
        _test_spec("rate-policy", trials=3, strategy="minPassRate", min_pass_rate=0.66),
    ]
    runner = AgentTestRunner(
        FakeAdapter(outcomes),
        {"schema": ResultEvaluator()},
        options=_options(tmp_path),
    )

    result = await runner.run(specs, _target())

    assert [case.passed for case in result.summary.cases] == [False, True, True]
    assert result.exit_code == 1
    assert result.summary.passed == 2
    assert result.summary.failed == 1
    assert result.summary.pass_rate == pytest.approx(2 / 3)
    for case in result.summary.cases:
        assert case.latency_ms.mean == pytest.approx(200)
        assert case.latency_ms.p50 == pytest.approx(200)
        assert case.latency_ms.p95 == pytest.approx(290)
        assert case.total_tokens.mean == pytest.approx(20)
        assert case.steps.p95 == pytest.approx(2.9)


async def test_prepare_failure_is_structured_and_cleanup_still_runs(tmp_path: Path) -> None:
    adapter = FakeAdapter([], prepare_error=RuntimeError("private prepare detail"))
    runner = AgentTestRunner(adapter, {"schema": ResultEvaluator()}, options=_options(tmp_path))

    result = await runner.run([_test_spec("prepare")], _target())

    assert result.status == "FAILED"
    assert result.exit_code == 2
    assert result.error is not None
    assert result.error.code == "TARGET_PREPARE_ERROR"
    assert "private prepare detail" not in result.error.message
    assert adapter.cleanup_calls == 1
    assert result.summary.incomplete == 1


async def test_prepare_preserves_trusted_adapter_diagnostic(tmp_path: Path) -> None:
    adapter = FakeAdapter([], prepare_error=TargetSafetyError("remote targets are disabled"))
    runner = AgentTestRunner(adapter, {"schema": ResultEvaluator()}, options=_options(tmp_path))

    result = await runner.run([_test_spec("unsafe-target")], _target())

    assert result.status == "FAILED"
    assert result.error is not None
    assert result.error.message == "remote targets are disabled"
    assert result.error.details["adapter_error_code"] == "HTTP_SAFETY_ERROR"


async def test_prepare_timeout_is_structured_and_cleanup_still_runs(tmp_path: Path) -> None:
    adapter = FakeAdapter([], prepare_delay_seconds=0.05)
    runner = AgentTestRunner(
        adapter,
        {"schema": ResultEvaluator()},
        options=_options(tmp_path, prepare_timeout_seconds=0.01),
    )

    result = await runner.run([_test_spec("prepare-timeout")], _target())

    assert result.status == "FAILED"
    assert result.error is not None
    assert result.error.code == "TARGET_PREPARE_ERROR"
    assert result.error.message == "target preparation timed out"
    assert adapter.cleanup_calls == 1


async def test_target_timeout_does_not_abort_later_trial(tmp_path: Path) -> None:
    adapter = FakeAdapter([Outcome(delay_seconds=0.05), Outcome(passed=True)])
    runner = AgentTestRunner(adapter, {"schema": ResultEvaluator()}, options=_options(tmp_path))

    result = await runner.run([_test_spec("timeout", trials=2)], _target(timeout_ms=10))

    assert result.status == "COMPLETED"
    assert [trial.target_result.status for trial in result.results] == ["TIMEOUT", "OK"]
    assert [trial.passed for trial in result.results] == [False, True]


async def test_evaluator_exception_becomes_error_result(tmp_path: Path) -> None:
    runner = AgentTestRunner(
        FakeAdapter([Outcome()]),
        {"schema": RaisingEvaluator()},
        options=_options(tmp_path),
    )

    result = await runner.run([_test_spec("evaluator-error")], _target())

    evaluation = result.results[0].evaluations[0]
    assert result.status == "COMPLETED"
    assert result.results[0].passed is False
    assert evaluation.status == "ERROR"
    assert evaluation.message is not None
    assert "sensitive evaluator detail" not in evaluation.message


async def test_evaluator_timeout_becomes_error_result(tmp_path: Path) -> None:
    runner = AgentTestRunner(
        FakeAdapter([Outcome()]),
        {"schema": SlowEvaluator()},
        options=_options(tmp_path, evaluator_timeout_seconds=0.01),
    )

    result = await runner.run([_test_spec("evaluator-timeout")], _target())

    evaluation = result.results[0].evaluations[0]
    assert result.status == "COMPLETED"
    assert evaluation.status == "ERROR"
    assert evaluation.message == "evaluator schema timed out"


async def test_trace_evaluator_is_not_observable_but_output_is_preserved(tmp_path: Path) -> None:
    runner = AgentTestRunner(
        FakeAdapter([Outcome()]),
        {"schema": TraceRequiredEvaluator()},
        options=_options(tmp_path),
    )

    result = await runner.run([_test_spec("missing-trace")], _target())

    trial = result.results[0]
    assert trial.target_result.status == "OK"
    assert trial.target_result.output == {"passed": True}
    assert trial.trace_status == "missing"
    assert trial.evaluations[0].status == "NOT_OBSERVABLE"
    trace_path = (
        tmp_path / "artifacts" / "runs" / result.run_id / "traces" / f"{trial.trace_id}.json"
    )
    assert trace_path.is_file()


async def test_langgraph_trial_persists_complete_parented_trace(tmp_path: Path) -> None:
    target = LangGraphTargetSpec.model_validate(
        {
            "metadata": {"name": "traced-graph"},
            "type": "langgraph",
            "entrypoint": "tests.fixtures.langgraph_targets:build_graph",
        }
    )
    spec = _test_spec("traced-graph").model_copy(update={"evaluators": []})
    runner = AgentTestRunner(
        LangGraphTargetAdapter(target),
        {},
        options=_options(tmp_path),
    )

    result = await runner.run([spec], target)

    trial = result.results[0]
    assert result.exit_code == 0
    assert trial.trace_status == "complete"
    trace_path = (
        tmp_path / "artifacts" / "runs" / result.run_id / "traces" / f"{trial.trace_id}.json"
    )
    trace_record = json.loads(trace_path.read_text(encoding="utf-8"))
    root = next(
        event
        for event in trace_record["events"]
        if event["attributes"].get("agentqa.runner.root") is True
    )
    graph_root = next(
        event
        for event in trace_record["events"]
        if event["kind"] == "AGENT" and event["span_id"] != root["span_id"]
    )
    assert graph_root["parent_span_id"] == root["span_id"]


async def test_runner_applies_langgraph_tool_mutation_and_persists_score(
    tmp_path: Path,
) -> None:
    target = LangGraphTargetSpec.model_validate(
        {
            "metadata": {"name": "mutation-graph"},
            "type": "langgraph",
            "entrypoint": "tests.fixtures.langgraph_targets:build_mutation_graph",
        }
    )
    mutation = MutationSpec.model_validate(
        {
            "metadata": {"name": "lookup-exception"},
            "target": {"kind": "tool", "name": "campus.lookup"},
            "operation": {"type": "tool_exception"},
        }
    )
    spec = _test_spec("mutated-graph").model_copy(update={"mutation": mutation.metadata.name})
    runner = AgentTestRunner(
        LangGraphTargetAdapter(target),
        {"schema": ResultEvaluator()},
        options=_options(tmp_path),
        mutations={mutation.metadata.name: mutation},
    )

    result = await runner.run([spec], target)

    assert result.exit_code == 1
    assert result.results[0].target_result.status == "ERROR"
    application = result.results[0].mutation_applications[0]
    assert application.mutation_id == "lookup-exception"
    assert application.triggered is True
    assert result.mutation_score is not None
    assert result.mutation_score.overall.killed == 1
    assert result.mutation_score.overall.score == 1.0
    root = tmp_path / "artifacts" / "runs" / result.run_id
    assert (root / "resolved-mutations.json").is_file()
    score = json.loads((root / "mutation-score.json").read_text(encoding="utf-8"))
    assert score["overall"]["killed"] == 1


async def test_cleanup_failure_fails_run_without_losing_results(tmp_path: Path) -> None:
    adapter = FakeAdapter([Outcome()], cleanup_error=RuntimeError("private cleanup detail"))
    runner = AgentTestRunner(adapter, {"schema": ResultEvaluator()}, options=_options(tmp_path))

    result = await runner.run([_test_spec("cleanup")], _target())

    assert len(result.results) == 1
    assert result.status == "FAILED"
    assert result.error is not None
    assert result.error.code == "TARGET_CLEANUP_ERROR"
    assert "private cleanup detail" not in result.error.message


async def test_cleanup_timeout_fails_run_without_losing_results(tmp_path: Path) -> None:
    adapter = FakeAdapter([Outcome()], cleanup_delay_seconds=0.05)
    runner = AgentTestRunner(
        adapter,
        {"schema": ResultEvaluator()},
        options=_options(tmp_path, cleanup_timeout_seconds=0.01),
    )

    result = await runner.run([_test_spec("cleanup-timeout")], _target())

    assert len(result.results) == 1
    assert result.status == "FAILED"
    assert result.error is not None
    assert result.error.code == "TARGET_CLEANUP_ERROR"
    assert result.error.message == "target cleanup timed out"


async def test_artifact_initialization_failure_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write_run(self: object, run: object) -> str:
        raise ArtifactError("synthetic artifact failure")

    monkeypatch.setattr("agentqa.storage.artifacts.ArtifactWriter.write_run", fail_write_run)
    runner = AgentTestRunner(
        FakeAdapter([]),
        {"schema": ResultEvaluator()},
        options=_options(tmp_path),
    )

    result = await runner.run([_test_spec("artifact-failure")], _target())

    assert result.status == "FAILED"
    assert result.error is not None
    assert result.error.code == "ARTIFACT_ERROR"


async def test_cooperative_cancel_stops_current_trial_and_finalizes(tmp_path: Path) -> None:
    adapter = BlockingAdapter()
    runner = AgentTestRunner(adapter, {"schema": ResultEvaluator()}, options=_options(tmp_path))
    task = asyncio.create_task(runner.run([_test_spec("cancelled")], _target(timeout_ms=10_000)))
    await asyncio.wait_for(adapter.started.wait(), timeout=1)

    runner.cancel()
    result = await asyncio.wait_for(task, timeout=1)

    assert result.status == "CANCELLED"
    assert result.exit_code == 2
    assert result.error is not None
    assert result.error.code == "USER_CANCELLED"
    assert result.results == []
    assert result.summary.incomplete == 1
    assert adapter.cleanup_calls == 1
    root = tmp_path / "artifacts" / "runs" / result.run_id
    assert (root / "summary.json").is_file()


async def test_trial_seeds_are_stable_for_the_same_run_seed(tmp_path: Path) -> None:
    spec = _test_spec("seeded", trials=2)
    first = AgentTestRunner(
        FakeAdapter([Outcome(), Outcome()]),
        {"schema": ResultEvaluator()},
        options=_options(tmp_path / "first"),
    )
    second = AgentTestRunner(
        FakeAdapter([Outcome(), Outcome()]),
        {"schema": ResultEvaluator()},
        options=_options(tmp_path / "second"),
    )

    first_result = await first.run([spec], _target())
    second_result = await second.run([spec], _target())

    assert first_result.exit_code == 0
    assert second_result.exit_code == 0
    assert [item.trial_seed for item in first_result.results] == [
        item.trial_seed for item in second_result.results
    ]
    assert len({item.trial_id for item in first_result.results}) == 2
    assert len({item.trace_id for item in first_result.results}) == 2


async def test_each_trial_receives_an_isolated_input_copy(tmp_path: Path) -> None:
    spec = _test_spec("isolated-input", trials=2)
    spec.scenario.input["nested"] = {"value": 0}
    adapter = InputMutationAdapter()
    runner = AgentTestRunner(
        adapter,
        {"schema": ResultEvaluator()},
        options=_options(tmp_path),
    )

    result = await runner.run([spec], _target())

    assert result.exit_code == 0
    assert adapter.observed_values == [0, 0]
    assert spec.scenario.input["nested"] == {"value": 0}
