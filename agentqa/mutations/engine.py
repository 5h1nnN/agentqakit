"""Runtime mutation interception, evidence, and trace emission."""

import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from opentelemetry import trace

from agentqa.conventions import utc_now
from agentqa.models import (
    ErrorInfo,
    InterceptionPoint,
    MutationApplication,
    MutationMatch,
    MutationSpec,
    TraceEvent,
)
from agentqa.mutations.context import MutationTrialSession, get_current_mutation_session
from agentqa.mutations.matcher import match_mutation
from agentqa.mutations.operators import (
    InjectedMutationError,
    InjectedMutationTimeout,
    Invocation,
    MutationOperator,
    OperatorOutcome,
    built_in_operators,
)
from agentqa.mutations.utils import safe_jsonable, summarize
from agentqa.tracing.context import record_trace_event


@dataclass(frozen=True, slots=True)
class _Attempt:
    session: MutationTrialSession
    spec: MutationSpec
    invocation: Invocation
    match: MutationMatch
    span_id: str
    parent_span_id: str | None
    started_at: Any


class MutationEngine:
    """Apply named MutationSpecs at wrapped runtime boundaries."""

    def __init__(
        self,
        specs: Mapping[str, MutationSpec] | None = None,
        operators: Mapping[str, MutationOperator] | None = None,
    ) -> None:
        self._specs = dict(specs or {})
        self._operators = dict(operators or built_in_operators())
        for name, spec in self._specs.items():
            if name != spec.metadata.name:
                raise ValueError("mutation mapping keys must equal metadata.name")
            operator = self._operators.get(spec.operation.type)
            if operator is None and spec.target.kind != "http":
                raise ValueError(f"no operator registered for {spec.operation.type}")

    @property
    def specs(self) -> Mapping[str, MutationSpec]:
        """Return the immutable-by-convention mutation registry."""
        return self._specs

    def _attempt(
        self,
        interception: InterceptionPoint,
        arguments: dict[str, Any],
    ) -> _Attempt | None:
        session = get_current_mutation_session()
        if session is None or session.run_context.mutation_id is None:
            return None
        spec = self._specs.get(session.run_context.mutation_id)
        if spec is None:
            return None
        call_index = session.next_call_index(interception.kind, interception.name)
        match = match_mutation(spec, interception, arguments, call_index, session)
        current = trace.get_current_span().get_span_context()
        parent_span_id = f"{current.span_id:016x}" if current.is_valid else None
        return _Attempt(
            session=session,
            spec=spec,
            invocation=Invocation(
                interception=interception,
                arguments=arguments,
                call_index=call_index,
                mutation_id=spec.metadata.name,
                operation=spec.operation,
                session=session,
            ),
            match=match,
            span_id=secrets.token_hex(8),
            parent_span_id=parent_span_id,
            started_at=utc_now(),
        )

    @staticmethod
    def _passive_cache(attempt: _Attempt, result: Any) -> None:
        operation = attempt.spec.operation.type
        key = attempt.invocation.cache_key
        if operation == "duplicate_call_result":
            attempt.session.caches.setdefault(key, result)
        elif operation == "stale_result":
            attempt.session.caches[key] = result

    @staticmethod
    def _record(
        attempt: _Attempt,
        *,
        result: Any = None,
        error: BaseException | None = None,
        triggered: bool,
        behavior: dict[str, Any],
    ) -> None:
        context = attempt.session.run_context
        if context.test_case_id is None or context.trial_id is None or context.trace_id is None:
            raise RuntimeError("mutation interception requires a complete trial context")
        safe_behavior = safe_jsonable(behavior)
        if not isinstance(safe_behavior, dict):
            safe_behavior = {"action": "unknown"}
        application = MutationApplication(
            mutation_id=attempt.spec.metadata.name,
            operation_type=attempt.spec.operation.type,
            interception=attempt.invocation.interception,
            call_index=attempt.invocation.call_index,
            triggered=triggered,
            match=attempt.match,
            invocation_summary=summarize(attempt.invocation.arguments),
            result_summary=None if error is not None else summarize(result),
            behavior=safe_behavior,
            test_case_id=context.test_case_id,
            trial_id=context.trial_id,
            trace_id=context.trace_id,
            span_id=attempt.span_id,
            applied_at=utc_now(),
        )
        attempt.session.applications.append(application)
        if triggered:
            attempt.session.applied_mutations.add(attempt.spec.metadata.name)

        error_info = None
        if error is not None:
            error_info = ErrorInfo(
                code=(
                    "MUTATION_INJECTED_ERROR"
                    if isinstance(error, InjectedMutationError | InjectedMutationTimeout)
                    else "INTERCEPTED_CALL_ERROR"
                ),
                message=(
                    "mutation deliberately raised an error"
                    if triggered
                    else "intercepted call raised an error"
                ),
                details={"exception_type": type(error).__name__},
            )
        kind: Literal["TOOL", "RETRIEVER"] = (
            "TOOL" if attempt.invocation.interception.kind == "tool" else "RETRIEVER"
        )
        record_trace_event(
            TraceEvent(
                trace_id=context.trace_id,
                span_id=attempt.span_id,
                parent_span_id=attempt.parent_span_id,
                kind=kind,
                name=(
                    f"agentqa.mutation.{attempt.invocation.interception.kind}."
                    f"{attempt.invocation.interception.name}"
                ),
                input=summarize(attempt.invocation.arguments),
                output=None if error is not None else summarize(result),
                started_at=attempt.started_at,
                ended_at=utc_now(),
                status="ERROR" if error is not None else "OK",
                error=error_info,
                attributes={
                    "agentqa.mutation.id": attempt.spec.metadata.name,
                    "agentqa.mutation.operation": attempt.spec.operation.type,
                    "agentqa.mutation.matched": attempt.match.matched,
                    "agentqa.mutation.triggered": triggered,
                    "agentqa.mutation.call_index": attempt.invocation.call_index,
                },
            )
        )

    async def intercept(
        self,
        interception: InterceptionPoint,
        arguments: dict[str, Any],
        next_call: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Apply an active mutation around one asynchronous invocation."""
        attempt = self._attempt(interception, arguments)
        if attempt is None:
            return await next_call()
        if not attempt.match.matched:
            try:
                result = await next_call()
            except BaseException as exc:
                self._record(
                    attempt,
                    error=exc,
                    triggered=False,
                    behavior={"action": "pass_through"},
                )
                raise
            self._passive_cache(attempt, result)
            self._record(
                attempt,
                result=result,
                triggered=False,
                behavior={"action": "pass_through"},
            )
            return result

        operator = self._operators[attempt.spec.operation.type]
        if not operator.supports(interception):
            raise RuntimeError("mutation operator does not support interception point")
        try:
            outcome = await operator.apply(attempt.invocation, next_call)
        except BaseException as exc:
            triggered = isinstance(exc, InjectedMutationError | InjectedMutationTimeout)
            self._record(
                attempt,
                error=exc,
                triggered=triggered,
                behavior={"action": "raise", "exception_type": type(exc).__name__},
            )
            raise
        self._record(
            attempt,
            result=outcome.value,
            triggered=outcome.triggered,
            behavior=outcome.behavior or {"action": "replace_result"},
        )
        return outcome.value

    def intercept_sync(
        self,
        interception: InterceptionPoint,
        arguments: dict[str, Any],
        next_call: Callable[[], Any],
    ) -> Any:
        """Apply an active mutation around one synchronous invocation."""
        attempt = self._attempt(interception, arguments)
        if attempt is None:
            return next_call()
        if not attempt.match.matched:
            try:
                result = next_call()
            except BaseException as exc:
                self._record(
                    attempt,
                    error=exc,
                    triggered=False,
                    behavior={"action": "pass_through"},
                )
                raise
            self._passive_cache(attempt, result)
            self._record(
                attempt,
                result=result,
                triggered=False,
                behavior={"action": "pass_through"},
            )
            return result

        operator = self._operators[attempt.spec.operation.type]
        if not operator.supports(interception):
            raise RuntimeError("mutation operator does not support interception point")
        try:
            outcome: OperatorOutcome = operator.apply_sync(attempt.invocation, next_call)
        except BaseException as exc:
            triggered = isinstance(exc, InjectedMutationError | InjectedMutationTimeout)
            self._record(
                attempt,
                error=exc,
                triggered=triggered,
                behavior={"action": "raise", "exception_type": type(exc).__name__},
            )
            raise
        self._record(
            attempt,
            result=outcome.value,
            triggered=outcome.triggered,
            behavior=outcome.behavior or {"action": "replace_result"},
        )
        return outcome.value
