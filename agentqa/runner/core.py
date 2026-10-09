"""Sequential, crash-aware AgentQAKit test runner."""

import asyncio
import hashlib
from collections.abc import Coroutine, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any, Literal, TypeVar, cast

from agentqa.conventions import new_id, to_utc_iso8601, utc_now
from agentqa.evaluators import EvaluationRequest, Evaluator
from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    ErrorInfo,
    EvaluationResult,
    MutationApplication,
    MutationScoreReport,
    MutationSpec,
    RunContext,
    RunResult,
    RunSummary,
    TargetResult,
    TargetSpec,
    TestInput,
    TestResult,
    TraceRecord,
    Violation,
)
from agentqa.models.specs import EvaluatorSpec
from agentqa.mutations import (
    MutationEngine,
    MutationTrialSession,
    calculate_mutation_score,
    reset_current_mutation_session,
    set_current_mutation_session,
)
from agentqa.runner.aggregation import aggregate_run
from agentqa.runner.context import reset_current_run_context, set_current_run_context
from agentqa.runner.errors import RunErrorCode
from agentqa.runner.options import RunnerOptions
from agentqa.runner.state import RunState, RunStateMachine
from agentqa.specs import hash_spec, hash_spec_collection, normalize_spec
from agentqa.storage import ArtifactError, ArtifactWriter
from agentqa.targets import TargetAdapter, TargetConfigurationError, TargetSafetyError
from agentqa.tracing import InMemoryTraceStore, TraceOptions, TraceRuntime, TraceStore
from agentqa.tracing.context import (
    reset_current_trace_options,
    reset_current_trace_store,
    set_current_trace_options,
    set_current_trace_store,
)
from agentqa.tracing.instrumentation import InstrumentationManager

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
        trace_store: TraceStore | None = None,
        trace_options: TraceOptions | None = None,
        mutations: Mapping[str, MutationSpec] | None = None,
        mutation_engine: MutationEngine | None = None,
    ) -> None:
        self._adapter = adapter
        self._evaluators = dict(evaluators)
        self._options = options or RunnerOptions()
        self._trace_options = trace_options or TraceOptions(
            flush_timeout_seconds=self._options.trace_flush_timeout_seconds,
            quiet_period_seconds=self._options.trace_quiet_period_seconds,
        )
        self._trace_store = trace_store or InMemoryTraceStore(
            max_spans_per_trace=self._trace_options.max_spans_per_trace
        )
        self._trace_runtime = TraceRuntime(self._trace_store, self._trace_options)
        if mutations is None and mutation_engine is not None:
            self._mutations = dict(mutation_engine.specs)
        else:
            self._mutations = dict(mutations or {})
        self._mutation_engine = mutation_engine or MutationEngine(self._mutations)
        if dict(self._mutation_engine.specs) != self._mutations:
            raise ValueError("mutation engine specs must match runner mutations")
        if self._mutations:
            configure_mutations = getattr(self._adapter, "set_mutation_engine", None)
            if callable(configure_mutations):
                configure_mutations(self._mutation_engine)
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
        test_spec: AgentTestSpec,
        target_result: TargetResult,
        trace_record: TraceRecord,
        context: RunContext,
    ) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for evaluator_spec in test_spec.evaluators:
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
            if target_result.status != "OK" and not getattr(
                evaluator, "runs_on_target_error", False
            ):
                results.append(
                    EvaluationResult(
                        evaluator_type=evaluator_spec.type,
                        status="SKIPPED",
                        message="target invocation did not complete",
                    )
                )
                continue
            if getattr(evaluator, "requires_trace", False) and trace_record.status != "complete":
                results.append(
                    EvaluationResult(
                        evaluator_type=evaluator_spec.type,
                        status="NOT_OBSERVABLE",
                        message=(
                            "trace evidence is unavailable or incomplete; "
                            "the target output remains available to output evaluators"
                        ),
                        metadata={"trace_status": trace_record.status},
                    )
                )
                continue
            try:
                result = await self._wait_or_cancel(
                    evaluator.evaluate(
                        EvaluationRequest(
                            evaluator_spec=evaluator_spec,
                            test_spec=test_spec,
                            target_result=target_result,
                            trace_record=trace_record,
                            context=context,
                        )
                    ),
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
        trace_record: TraceRecord,
        mutation_applications: Sequence[MutationApplication],
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
            trace_status=trace_record.status,
            trace_ref=f"traces/{context.trace_id}.json",
            mutation_applications=list(mutation_applications),
        )

    async def _execute_trial(
        self,
        spec: AgentTestSpec,
        target: TargetSpec,
        base_context: RunContext,
        trial_index: int,
    ) -> tuple[TestResult, TraceRecord]:
        started_at = utc_now()
        context = base_context.model_copy(
            update={
                "test_case_id": spec.metadata.name,
                "trial_id": new_id(),
                "trace_id": new_id(),
                "mutation_id": spec.mutation,
                "test_labels": spec.metadata.labels,
                "trial_seed": _derive_trial_seed(
                    base_context.random_seed,
                    spec.metadata.name,
                    trial_index,
                ),
            }
        )
        token = set_current_run_context(context)
        trace_token = set_current_trace_store(self._trace_store)
        trace_options_token = set_current_trace_options(self._trace_options)
        mutation_session = MutationTrialSession(context)
        mutation_token = set_current_mutation_session(mutation_session)
        try:
            try:
                async with asyncio.timeout(self._options.item_timeout_seconds):
                    with self._trace_runtime.trial_span(context) as root_span:
                        target_result = await self._invoke_target(spec, target, context)
                        self._trace_runtime.finish_target_span(root_span, target_result)
                    await self._trace_store.wait_for_quiet(
                        context.trace_id or "",
                        timeout_seconds=self._trace_options.flush_timeout_seconds,
                        quiet_period_seconds=self._trace_options.quiet_period_seconds,
                    )
                    trace_record = self._trace_store.build_record(context.trace_id or "")
                    evaluations = await self._evaluate(
                        spec,
                        target_result,
                        trace_record,
                        context,
                    )
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
                trace_record = self._trace_store.build_record(context.trace_id or "")
            result = self._build_test_result(
                spec=spec,
                context=context,
                trial_index=trial_index,
                started_at=started_at,
                target_result=target_result,
                evaluations=evaluations,
                trace_record=trace_record,
                mutation_applications=mutation_session.applications,
            )
            return result, trace_record
        finally:
            reset_current_mutation_session(mutation_token)
            reset_current_trace_options(trace_options_token)
            reset_current_trace_store(trace_token)
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

    def _validate_specs(self, specs: Sequence[AgentTestSpec]) -> ErrorInfo | None:
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
        missing_mutations = sorted(
            {
                spec.mutation
                for spec in specs
                if spec.mutation is not None and spec.mutation not in self._mutations
            }
        )
        if missing_mutations:
            return ErrorInfo(
                code=RunErrorCode.SPEC_ERROR.value,
                message=(
                    "AgentTest references unknown mutation(s): " + ", ".join(missing_mutations)
                ),
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
        hashes.extend(hash_spec(spec) for _, spec in sorted(self._mutations.items()))
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
            otlp_endpoint=self._options.otlp_endpoint,
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
        instrumentation = InstrumentationManager()
        if target.type == "langgraph":
            instrumentation.activate(
                ["langgraph", "openai"],
                self._trace_runtime.provider,
                self._trace_options,
            )

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
            if self._mutations:
                writer.write_resolved_mutations(
                    [normalize_spec(spec) for _, spec in sorted(self._mutations.items())]
                )
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
                except (TargetConfigurationError, TargetSafetyError) as exc:
                    raise _FatalRunError(
                        ErrorInfo(
                            code=RunErrorCode.TARGET_PREPARE_ERROR.value,
                            message=exc.message,
                            details={"adapter_error_code": exc.code},
                        )
                    ) from None
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
                        result, trace_record = await self._execute_trial(
                            spec, target, context, trial_index
                        )
                        results.append(result)
                        writer.write_trace(result.trace_id, trace_record)
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
        mutation_score: MutationScoreReport | None = None
        if self._mutations:
            mutation_score = calculate_mutation_score(
                self._mutations,
                frozen_specs,
                results,
            )
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
                if mutation_score is not None:
                    writer.write_mutation_score(mutation_score)
                writer.finalize_summary(
                    {
                        "run_id": run_id,
                        "status": desired_state.value,
                        "exit_code": exit_code,
                        "summary": summary.model_dump(mode="json"),
                        "mutation_score": (
                            None
                            if mutation_score is None
                            else mutation_score.model_dump(mode="json")
                        ),
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
            mutation_score=mutation_score,
            error=run_error,
        )
        instrumentation.close()
        self._trace_runtime.shutdown()
        if external_cancellation:
            raise asyncio.CancelledError
        return run_result
