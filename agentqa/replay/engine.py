"""Record/replay interception and composition with the mutation engine."""

import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from opentelemetry import trace

from agentqa.conventions import utc_now
from agentqa.errors import AgentQAError
from agentqa.models import ErrorInfo, InterceptionPoint, TraceEvent
from agentqa.mutations.engine import MutationEngine
from agentqa.mutations.utils import summarize
from agentqa.replay.context import ReplaySession, get_current_replay_session
from agentqa.replay.models import CassetteEntry, CassetteError
from agentqa.replay.normalization import normalize_arguments, safe_value
from agentqa.replay.store import ReplayError
from agentqa.storage import redact_text_secrets
from agentqa.tracing.context import record_trace_event


class CassetteMissError(ReplayError):
    """Raised when strict replay has no matching interaction."""

    def __init__(self, kind: str, name: str, call_index: int) -> None:
        super().__init__(
            f"cassette miss for {kind} {name!r} call {call_index}",
            code="CASSETTE_MISS",
        )


class ReplayedDependencyError(ReplayError):
    """Safe equivalent of an exception captured during recording."""

    def __init__(self, error: CassetteError) -> None:
        super().__init__(
            f"replayed dependency error {error.exception_type}",
            code=error.code,
        )
        self.original_exception_type = error.exception_type


@dataclass(frozen=True, slots=True)
class _Attempt:
    session: ReplaySession
    kind: Literal["tool", "retriever", "http"]
    name: str
    normalized_arguments: Any
    call_index: int
    trace_id: str
    span_id: str
    parent_span_id: str | None
    started_at: Any


def _attempt(interception: InterceptionPoint) -> _Attempt | None:
    session = get_current_replay_session()
    if session is None:
        return None
    context = session.run_context
    if context.trace_id is None:
        raise RuntimeError("replay interception requires a trace ID")
    current = trace.get_current_span().get_span_context()
    parent_span_id = f"{current.span_id:016x}" if current.is_valid else None
    return _Attempt(
        session=session,
        kind=interception.kind,
        name=interception.name,
        normalized_arguments=None,
        call_index=session.next_call_index(interception.kind, interception.name),
        trace_id=context.trace_id,
        span_id=secrets.token_hex(8),
        parent_span_id=parent_span_id,
        started_at=utc_now(),
    )


def _safe_error(exc: Exception) -> CassetteError:
    code = exc.code if isinstance(exc, AgentQAError) else "DEPENDENCY_ERROR"
    return CassetteError(
        code=code,
        exception_type=type(exc).__name__,
        message=redact_text_secrets(f"dependency raised {type(exc).__name__}"),
    )


def _trace_attempt(
    attempt: _Attempt,
    *,
    source: Literal["live", "record", "cassette", "live_fallback", "miss"],
    result: Any = None,
    error: CassetteError | None = None,
) -> None:
    kind: Literal["TOOL", "RETRIEVER"] = "TOOL" if attempt.kind == "tool" else "RETRIEVER"
    record_trace_event(
        TraceEvent(
            trace_id=attempt.trace_id,
            span_id=attempt.span_id,
            parent_span_id=attempt.parent_span_id,
            kind=kind,
            name=f"agentqa.replay.{attempt.kind}.{attempt.name}",
            input=summarize(attempt.normalized_arguments),
            output=None if error is not None else summarize(result),
            started_at=attempt.started_at,
            ended_at=utc_now(),
            status="ERROR" if error is not None else "OK",
            error=(
                None
                if error is None
                else ErrorInfo(
                    code=error.code,
                    message=error.message,
                    details={"exception_type": error.exception_type},
                )
            ),
            attributes={
                "agentqa.replay.source": source,
                "agentqa.replay.call_index": attempt.call_index,
                "agentqa.interception.name": attempt.name,
            },
        )
    )


def _prepare_attempt(
    interception: InterceptionPoint,
    arguments: dict[str, Any],
) -> _Attempt | None:
    attempt = _attempt(interception)
    if attempt is None:
        return None
    normalized = normalize_arguments(arguments, attempt.session.volatile_fields)
    return _Attempt(
        session=attempt.session,
        kind=attempt.kind,
        name=attempt.name,
        normalized_arguments=normalized,
        call_index=attempt.call_index,
        trace_id=attempt.trace_id,
        span_id=attempt.span_id,
        parent_span_id=attempt.parent_span_id,
        started_at=attempt.started_at,
    )


def _record_success(attempt: _Attempt, result: Any) -> Any:
    safe_result = safe_value(result)
    attempt.session.store.record_success(
        kind=attempt.kind,
        name=attempt.name,
        arguments=attempt.normalized_arguments,
        call_index=attempt.call_index,
        result=safe_result,
        trace_id=attempt.trace_id,
        span_id=attempt.span_id,
    )
    attempt.session.note_recorded()
    return result


def _record_error(attempt: _Attempt, error: CassetteError) -> None:
    attempt.session.store.record_error(
        kind=attempt.kind,
        name=attempt.name,
        arguments=attempt.normalized_arguments,
        call_index=attempt.call_index,
        error=error,
        trace_id=attempt.trace_id,
        span_id=attempt.span_id,
    )
    attempt.session.note_recorded()


def _replay(attempt: _Attempt) -> CassetteEntry | None:
    return attempt.session.store.lookup(
        attempt.kind,
        attempt.name,
        attempt.normalized_arguments,
        attempt.call_index,
    )


class ReplayEngine:
    """Apply the current trial's strict record/replay policy."""

    async def intercept(
        self,
        interception: InterceptionPoint,
        arguments: dict[str, Any],
        next_call: Callable[[], Awaitable[Any]],
    ) -> Any:
        attempt = _prepare_attempt(interception, arguments)
        if attempt is None:
            return await next_call()
        if attempt.session.mode == "off":
            try:
                result = await next_call()
            except Exception as exc:
                _trace_attempt(attempt, source="live", error=_safe_error(exc))
                raise
            _trace_attempt(attempt, source="live", result=result)
            return result
        if attempt.session.mode == "record":
            try:
                result = await next_call()
            except Exception as exc:
                error = _safe_error(exc)
                _record_error(attempt, error)
                _trace_attempt(attempt, source="record", error=error)
                raise
            _record_success(attempt, result)
            _trace_attempt(attempt, source="record", result=result)
            return result

        entry = _replay(attempt)
        if entry is not None:
            attempt.session.note_hit()
            if entry.status == "error":
                assert entry.error is not None
                _record_error(attempt, entry.error)
                _trace_attempt(attempt, source="cassette", error=entry.error)
                raise ReplayedDependencyError(entry.error)
            result = attempt.session.store.result(entry)
            _record_success(attempt, result)
            _trace_attempt(attempt, source="cassette", result=result)
            return result

        attempt.session.note_miss()
        if not attempt.session.allow_live_fallback:
            error = CassetteError(
                code="CASSETTE_MISS",
                exception_type="CassetteMissError",
                message="strict replay had no matching cassette entry",
            )
            _trace_attempt(attempt, source="miss", error=error)
            raise CassetteMissError(attempt.kind, attempt.name, attempt.call_index)
        attempt.session.note_live_fallback()
        try:
            result = await next_call()
        except Exception as exc:
            error = _safe_error(exc)
            _record_error(attempt, error)
            _trace_attempt(attempt, source="live_fallback", error=error)
            raise
        _record_success(attempt, result)
        _trace_attempt(attempt, source="live_fallback", result=result)
        return result

    def intercept_sync(
        self,
        interception: InterceptionPoint,
        arguments: dict[str, Any],
        next_call: Callable[[], Any],
    ) -> Any:
        attempt = _prepare_attempt(interception, arguments)
        if attempt is None:
            return next_call()
        if attempt.session.mode == "off":
            try:
                result = next_call()
            except Exception as exc:
                _trace_attempt(attempt, source="live", error=_safe_error(exc))
                raise
            _trace_attempt(attempt, source="live", result=result)
            return result
        if attempt.session.mode == "record":
            try:
                result = next_call()
            except Exception as exc:
                error = _safe_error(exc)
                _record_error(attempt, error)
                _trace_attempt(attempt, source="record", error=error)
                raise
            _record_success(attempt, result)
            _trace_attempt(attempt, source="record", result=result)
            return result

        entry = _replay(attempt)
        if entry is not None:
            attempt.session.note_hit()
            if entry.status == "error":
                assert entry.error is not None
                _record_error(attempt, entry.error)
                _trace_attempt(attempt, source="cassette", error=entry.error)
                raise ReplayedDependencyError(entry.error)
            result = attempt.session.store.result(entry)
            _record_success(attempt, result)
            _trace_attempt(attempt, source="cassette", result=result)
            return result

        attempt.session.note_miss()
        if not attempt.session.allow_live_fallback:
            error = CassetteError(
                code="CASSETTE_MISS",
                exception_type="CassetteMissError",
                message="strict replay had no matching cassette entry",
            )
            _trace_attempt(attempt, source="miss", error=error)
            raise CassetteMissError(attempt.kind, attempt.name, attempt.call_index)
        attempt.session.note_live_fallback()
        try:
            result = next_call()
        except Exception as exc:
            error = _safe_error(exc)
            _record_error(attempt, error)
            _trace_attempt(attempt, source="live_fallback", error=error)
            raise
        _record_success(attempt, result)
        _trace_attempt(attempt, source="live_fallback", result=result)
        return result


class InteractionEngine:
    """Apply mutations outside record/replay so cassettes retain baseline data."""

    def __init__(self, mutation_engine: MutationEngine | None = None) -> None:
        self._mutation_engine = mutation_engine
        self._replay_engine = ReplayEngine()

    async def intercept(
        self,
        interception: InterceptionPoint,
        arguments: dict[str, Any],
        next_call: Callable[[], Awaitable[Any]],
    ) -> Any:
        async def replay_call() -> Any:
            return await self._replay_engine.intercept(
                interception,
                arguments,
                next_call,
            )

        if self._mutation_engine is None:
            return await replay_call()
        return await self._mutation_engine.intercept(interception, arguments, replay_call)

    def intercept_sync(
        self,
        interception: InterceptionPoint,
        arguments: dict[str, Any],
        next_call: Callable[[], Any],
    ) -> Any:
        def replay_call() -> Any:
            return self._replay_engine.intercept_sync(
                interception,
                arguments,
                next_call,
            )

        if self._mutation_engine is None:
            return replay_call()
        return self._mutation_engine.intercept_sync(interception, arguments, replay_call)
