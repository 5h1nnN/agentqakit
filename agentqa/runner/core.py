"""Sequential, crash-aware AgentQAKit test runner."""

import asyncio
import hashlib
from collections.abc import Coroutine, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any, Literal, TypeVar, cast

from agentqa.conventions import new_id, to_utc_iso8601, utc_now
from agentqa.evaluators import Evaluator
from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    ErrorInfo,
    EvaluationResult,
    RunContext,
    RunResult,
    RunSummary,
    TargetResult,
    TargetSpec,
    TestInput,
    TestResult,
    Violation,
)
from agentqa.models.specs import EvaluatorSpec
from agentqa.runner.aggregation import aggregate_run
from agentqa.runner.context import reset_current_run_context, set_current_run_context
from agentqa.runner.errors import RunErrorCode
from agentqa.runner.options import RunnerOptions
from agentqa.runner.state import RunState, RunStateMachine
from agentqa.specs import hash_spec, hash_spec_collection, normalize_spec
from agentqa.storage import ArtifactError, ArtifactWriter
from agentqa.targets import TargetAdapter

_T = TypeVar("_T")


class _CancellationRequested(Exception):
    pass


class _FatalRunError(Exception):
    def __init__(self, error: ErrorInfo) -> None:
        super().__init__(error.message)
        self.error = error


def _safe_exception_message(subject: str, exc: BaseException) -> str:
    """Describe an exception without persisting its potentially sensitive message."""
    return f"{subject} raised {type(exc).__name__}"


def _derive_trial_seed(run_seed: int, test_case_id: str, trial_index: int) -> int:
    material = f"{run_seed}\0{test_case_id}\0{trial_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _merge_cleanup_error(current: ErrorInfo | None, cleanup: ErrorInfo) -> ErrorInfo:
    if current is None:
        return cleanup
    return ErrorInfo(
        code=current.code,
        message=current.message,
        details={
            **current.details,
            "cleanup_error": cleanup.model_dump(mode="json"),
        },
    )


def _run_exit_code(state: RunState, summary: RunSummary) -> Literal[0, 1, 2]:
    if state != RunState.COMPLETED or summary.incomplete > 0:
        return 2
    return 0 if summary.failed == 0 else 1


class TestRunner:
    """Run validated specs sequentially against injected adapters and evaluators."""

    def __init__(
        self,
        adapter: TargetAdapter,
        evaluators: Mapping[str, Evaluator],
        *,
        options: RunnerOptions | None = None,
    ) -> None:
        self._adapter = adapter
        self._evaluators = dict(evaluators)
        self._options = options or RunnerOptions()
        self._cancel_requested = asyncio.Event()
        self._has_run = False

    def cancel(self) -> None:
        """Request cooperative cancellation of the active run."""
        self._cancel_requested.set()

    async def _wait_or_cancel(
        self,
        operation: Coroutine[Any, Any, _T],
        *,
        timeout_seconds: float,
    ) -> _T:
        if self._cancel_requested.is_set():
            operation.close()
            raise _CancellationRequested
        operation_task = asyncio.create_task(operation)
        cancellation_task = asyncio.create_task(self._cancel_requested.wait())
        try:
            done, _ = await asyncio.wait(
                {operation_task, cancellation_task},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_task in done and self._cancel_requested.is_set():
                operation_task.cancel()
                await asyncio.gather(operation_task, return_exceptions=True)
                raise _CancellationRequested
            if operation_task in done:
                return operation_task.result()
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise TimeoutError
        finally:
            if not operation_task.done():
                operation_task.cancel()
                await asyncio.gather(operation_task, return_exceptions=True)
            cancellation_task.cancel()
            await asyncio.gather(cancellation_task, return_exceptions=True)

    @staticmethod
    def _test_input(spec: AgentTestSpec) -> TestInput:
        return TestInput(
            test_case_id=spec.metadata.name,
            input=deepcopy(spec.scenario.input),
            initial_state=deepcopy(spec.scenario.initial_state),
            messages=[message.model_dump(mode="json") for message in spec.scenario.messages],
            mutation=spec.mutation,
        )

    async def _invoke_target(
        self,
        spec: AgentTestSpec,
        target: TargetSpec,
        context: RunContext,
    ) -> TargetResult:
        started_at = utc_now()
        try:
            return await self._wait_or_cancel(
                self._adapter.invoke(self._test_input(spec), context),
                timeout_seconds=target.timeout_ms / 1000,
            )
        except _CancellationRequested:
            raise
        except TimeoutError:
            return TargetResult(
                status="TIMEOUT",
                error=ErrorInfo(
                    code=RunErrorCode.TARGET_TIMEOUT.value,
                    message="target invocation timed out",
                ),
                started_at=started_at,
                ended_at=utc_now(),
            )
        except Exception as exc:
            return TargetResult(
                status="ERROR",
                error=ErrorInfo(
                    code=RunErrorCode.TARGET_RUNTIME_ERROR.value,
                    message=_safe_exception_message("target adapter", exc),
                ),
                started_at=started_at,
                ended_at=utc_now(),
            )

    @staticmethod
    def _skipped_evaluations(
        specs: Sequence[EvaluatorSpec],
        message: str,
    ) -> list[EvaluationResult]:
        return [
            EvaluationResult(evaluator_type=spec.type, status="SKIPPED", message=message)
            for spec in specs
        ]

    async def _evaluate(
        self,
        specs: Sequence[EvaluatorSpec],
        target_result: TargetResult,
        context: RunContext,
    ) -> list[EvaluationResult]:
        if target_result.status != "OK":
            return self._skipped_evaluations(specs, "target invocation did not complete")

        results: list[EvaluationResult] = []
        for evaluator_spec in specs:
            evaluator = self._evaluators.get(evaluator_spec.type)
            if evaluator is None:
                results.append(
                    EvaluationResult(
                        evaluator_type=evaluator_spec.type,
                        status="ERROR",
                        message=f"no evaluator registered for {evaluator_spec.type}",
                        metadata={"error_code": RunErrorCode.EVALUATION_ERROR.value},
                    )
                )
                continue
            try:
                result = await self._wait_or_cancel(
                    evaluator.evaluate(evaluator_spec, target_result, None, context),
                    timeout_seconds=self._options.evaluator_timeout_seconds,
                )
            except _CancellationRequested:
                raise
            except TimeoutError:
                result = EvaluationResult(
                    evaluator_type=evaluator_spec.type,
                    status="ERROR",
                    message=f"evaluator {evaluator_spec.type} timed out",
                    metadata={"error_code": RunErrorCode.EVALUATION_ERROR.value},
                )
            except Exception as exc:
                result = EvaluationResult(
                    evaluator_type=evaluator_spec.type,
                    status="ERROR",
                    message=_safe_exception_message(f"evaluator {evaluator_spec.type}", exc),
                    metadata={"error_code": RunErrorCode.EVALUATION_ERROR.value},
                )
            if not isinstance(result, EvaluationResult):
                result = EvaluationResult(
                    evaluator_type=evaluator_spec.type,
                    status="ERROR",
                    message="evaluator returned an invalid result object",
                    metadata={"error_code": RunErrorCode.EVALUATION_ERROR.value},
                )
            elif result.evaluator_type != evaluator_spec.type:
                result = EvaluationResult(
                    evaluator_type=evaluator_spec.type,
                    status="ERROR",
                    message="evaluator returned a mismatched evaluator_type",
                    metadata={"error_code": RunErrorCode.EVALUATION_ERROR.value},
                )
            results.append(result)
        return results

    @staticmethod
    def _build_test_result(
        *,
        spec: AgentTestSpec,
        context: RunContext,
        trial_index: int,
        started_at: datetime,
        target_result: TargetResult,
        evaluations: Sequence[EvaluationResult],
    ) -> TestResult:
        if context.trial_id is None or context.trace_id is None or context.trial_seed is None:
            raise RuntimeError("trial context is incomplete")
        violations = [violation for result in evaluations for violation in result.violations]
        if target_result.error is not None:
            violations.insert(
                0,
                Violation(
                    type=target_result.error.code,
                    message=target_result.error.message,
                ),
            )
        scores = {
            result.evaluator_type: result.score
            for result in evaluations
            if result.score is not None
        }
        passed = target_result.status == "OK" and all(
            result.status == "PASS" for result in evaluations
        )
        return TestResult(
            test_case_id=spec.metadata.name,
            trial_id=context.trial_id,
            trace_id=context.trace_id,
            trial_index=trial_index,
            trial_seed=context.trial_seed,
            started_at=started_at,
            ended_at=utc_now(),
            passed=passed,
            scores=scores,
            violations=violations,
            evaluations=list(evaluations),
            target_result=target_result,
        )

    async def _execute_trial(
        self,
        spec: AgentTestSpec,
        target: TargetSpec,
        base_context: RunContext,
        trial_index: int,
    ) -> TestResult:
        started_at = utc_now()
        context = base_context.model_copy(
            update={
                "test_case_id": spec.metadata.name,
                "trial_id": new_id(),
                "trace_id": new_id(),
                "mutation_id": spec.mutation,
                "trial_seed": _derive_trial_seed(
                    base_context.random_seed,
                    spec.metadata.name,
                    trial_index,
                ),
            }
        )
        token = set_current_run_context(context)
        try:
            try:
                async with asyncio.timeout(self._options.item_timeout_seconds):
                    target_result = await self._invoke_target(spec, target, context)
                    evaluations = await self._evaluate(spec.evaluators, target_result, context)
            except TimeoutError:
                target_result = TargetResult(
                    status="TIMEOUT",
                    error=ErrorInfo(
                        code=RunErrorCode.TARGET_TIMEOUT.value,
                        message="test item timed out",
                    ),
                    started_at=started_at,
                    ended_at=utc_now(),
                )
                evaluations = self._skipped_evaluations(
                    spec.evaluators,
                    "test item exceeded its total timeout",
                )
            return self._build_test_result(
                spec=spec,
                context=context,
                trial_index=trial_index,
                started_at=started_at,
                target_result=target_result,
                evaluations=evaluations,
            )
        finally:
            reset_current_run_context(token)

    @staticmethod
    def _state_record(
        machine: RunStateMachine,
        context: RunContext,
        *,
        total_items: int,
        completed_items: int,
        error: ErrorInfo | None,
    ) -> dict[str, Any]:
        return {
            "run_id": context.run_id,
            "status": machine.state.value,
            "started_at": to_utc_iso8601(context.started_at),
            "updated_at": to_utc_iso8601(machine.history[-1].at),
            "spec_hash": context.spec_hash,
            "manifest_hash": context.manifest_hash,
            "random_seed": context.random_seed,
            "total_items": total_items,
            "completed_items": completed_items,
            "error": None if error is None else error.model_dump(mode="json"),
            "history": [
                {
                    "previous": None if item.previous is None else item.previous.value,
                    "current": item.current.value,
                    "at": to_utc_iso8601(item.at),
                }
                for item in machine.history
            ],
        }

    @staticmethod
    def _validate_specs(specs: Sequence[AgentTestSpec]) -> ErrorInfo | None:
        if not specs:
            return ErrorInfo(
                code=RunErrorCode.SPEC_ERROR.value,
                message="at least one AgentTest spec is required",
            )
        names = [spec.metadata.name for spec in specs]
        if len(names) != len(set(names)):
            return ErrorInfo(
                code=RunErrorCode.SPEC_ERROR.value,
                message="test case names must be unique within a run",
            )
        return None

    async def run(
        self,
        specs: Sequence[AgentTestSpec],
        target: TargetSpec,
        *,
        manifest: AgentManifest | None = None,
    ) -> RunResult:
        """Execute every ``Test x Trial`` item sequentially and finalize artifacts."""
        if self._has_run:
            raise RuntimeError("TestRunner instances are single-use")
        self._has_run = True
        frozen_specs = list(specs)
        started_at = utc_now()
        run_id = new_id()
        hashes = [hash_spec(target), *(hash_spec(spec) for spec in frozen_specs)]
        manifest_hash = None
        if manifest is not None:
            manifest_hash = hash_spec(manifest)
            hashes.append(manifest_hash)
        context = RunContext(
            run_id=run_id,
            spec_hash=hash_spec_collection(hashes),
            manifest_hash=manifest_hash,
            started_at=started_at,
            random_seed=self._options.random_seed,
            artifact_dir=str((self._options.artifact_directory / "runs" / run_id).resolve()),
        )
        writer = ArtifactWriter(self._options.artifact_directory, run_id)
        machine = RunStateMachine()
        results: list[TestResult] = []
        total_items = sum(spec.trials for spec in frozen_specs)
        run_error = self._validate_specs(frozen_specs)
        desired_state = RunState.FAILED if run_error is not None else RunState.COMPLETED
        prepare_attempted = False
        artifact_healthy = True
        external_cancellation = False

        try:
            writer.initialize(
                self._state_record(
                    machine,
                    context,
                    total_items=total_items,
                    completed_items=0,
                    error=run_error,
                )
            )
            writer.write_resolved_target(normalize_spec(target))
            writer.write_resolved_tests([normalize_spec(spec) for spec in frozen_specs])
        except ArtifactError:
            artifact_healthy = False
            run_error = ErrorInfo(
                code=RunErrorCode.ARTIFACT_ERROR.value,
                message="failed to initialize run artifacts",
            )
            desired_state = RunState.FAILED

        if artifact_healthy and run_error is None:
            try:
                machine.transition(RunState.PREPARING)
                writer.write_run(
                    self._state_record(
                        machine,
                        context,
                        total_items=total_items,
                        completed_items=0,
                        error=None,
                    )
                )
                prepare_attempted = True
                try:
                    await self._wait_or_cancel(
                        self._adapter.prepare(context),
                        timeout_seconds=self._options.prepare_timeout_seconds,
                    )
                except TimeoutError:
                    raise _FatalRunError(
                        ErrorInfo(
                            code=RunErrorCode.TARGET_PREPARE_ERROR.value,
                            message="target preparation timed out",
                        )
                    ) from None
                except _CancellationRequested:
                    raise
                except Exception as exc:
                    raise _FatalRunError(
                        ErrorInfo(
                            code=RunErrorCode.TARGET_PREPARE_ERROR.value,
                            message=_safe_exception_message("target preparation", exc),
                        )
                    ) from None

                machine.transition(RunState.RUNNING)
                writer.write_run(
                    self._state_record(
                        machine,
                        context,
                        total_items=total_items,
                        completed_items=0,
                        error=None,
                    )
                )
                for spec in frozen_specs:
                    for trial_index in range(1, spec.trials + 1):
                        if self._cancel_requested.is_set():
                            raise _CancellationRequested
                        result = await self._execute_trial(spec, target, context, trial_index)
                        results.append(result)
                        writer.append_result(result)
                        writer.write_run(
                            self._state_record(
                                machine,
                                context,
                                total_items=total_items,
                                completed_items=len(results),
                                error=None,
                            )
                        )

                machine.transition(RunState.EVALUATING)
                writer.write_run(
                    self._state_record(
                        machine,
                        context,
                        total_items=total_items,
                        completed_items=len(results),
                        error=None,
                    )
                )
            except _CancellationRequested:
                desired_state = RunState.CANCELLED
                run_error = ErrorInfo(
                    code=RunErrorCode.USER_CANCELLED.value,
                    message="run cancelled by user",
                )
            except _FatalRunError as exc:
                desired_state = RunState.FAILED
                run_error = exc.error
            except ArtifactError:
                artifact_healthy = False
                desired_state = RunState.FAILED
                run_error = ErrorInfo(
                    code=RunErrorCode.ARTIFACT_ERROR.value,
                    message="failed to persist run artifacts",
                )
            except asyncio.CancelledError:
                external_cancellation = True
                desired_state = RunState.CANCELLED
                run_error = ErrorInfo(
                    code=RunErrorCode.USER_CANCELLED.value,
                    message="run task was cancelled",
                )
            except Exception as exc:
                desired_state = RunState.FAILED
                run_error = ErrorInfo(
                    code=RunErrorCode.RUNNER_ERROR.value,
                    message=_safe_exception_message("test runner", exc),
                )

        if machine.state not in {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}:
            try:
                machine.transition(RunState.FINALIZING)
                if artifact_healthy:
                    writer.write_run(
                        self._state_record(
                            machine,
                            context,
                            total_items=total_items,
                            completed_items=len(results),
                            error=run_error,
                        )
                    )
            except ArtifactError:
                artifact_healthy = False
                desired_state = RunState.FAILED
                run_error = ErrorInfo(
                    code=RunErrorCode.ARTIFACT_ERROR.value,
                    message="failed to persist final state",
                )

        if prepare_attempted:
            try:
                await asyncio.wait_for(
                    self._adapter.cleanup(context),
                    timeout=self._options.cleanup_timeout_seconds,
                )
            except TimeoutError:
                desired_state = RunState.FAILED
                run_error = _merge_cleanup_error(
                    run_error,
                    ErrorInfo(
                        code=RunErrorCode.TARGET_CLEANUP_ERROR.value,
                        message="target cleanup timed out",
                    ),
                )
            except Exception as exc:
                desired_state = RunState.FAILED
                run_error = _merge_cleanup_error(
                    run_error,
                    ErrorInfo(
                        code=RunErrorCode.TARGET_CLEANUP_ERROR.value,
                        message=_safe_exception_message("target cleanup", exc),
                    ),
                )

        summary = aggregate_run(frozen_specs, results)
        ended_at = utc_now()
        if machine.state == RunState.FINALIZING:
            machine.transition(desired_state)
        exit_code = _run_exit_code(desired_state, summary)
        if artifact_healthy:
            try:
                writer.write_run(
                    self._state_record(
                        machine,
                        context,
                        total_items=total_items,
                        completed_items=len(results),
                        error=run_error,
                    )
                )
                writer.finalize_summary(
                    {
                        "run_id": run_id,
                        "status": desired_state.value,
                        "exit_code": exit_code,
                        "summary": summary.model_dump(mode="json"),
                    }
                )
            except ArtifactError:
                desired_state = RunState.FAILED
                run_error = ErrorInfo(
                    code=RunErrorCode.ARTIFACT_ERROR.value,
                    message="failed to finalize run artifacts",
                )
                ended_at = utc_now()
                exit_code = 2

        run_result = RunResult(
            run_id=run_id,
            status=cast(Literal["COMPLETED", "FAILED", "CANCELLED"], desired_state.value),
            exit_code=exit_code,
            started_at=started_at,
            ended_at=ended_at,
            summary=summary,
            results=results,
            error=run_error,
        )
        if external_cancellation:
            raise asyncio.CancelledError
        return run_result
