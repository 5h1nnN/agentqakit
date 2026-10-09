"""Thread-safe in-memory and JSONL trace stores."""

import asyncio
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from agentqa.conventions import to_utc_iso8601, utc_now
from agentqa.models import TraceEvent, TraceRecord
from agentqa.specs.hashing import canonical_json_bytes
from agentqa.storage import redact_secrets

_RUNNER_ROOT_ATTRIBUTE = "agentqa.runner.root"


class TraceStore(Protocol):
    """Collect, query, and assess normalized trace events."""

    def add(self, event: TraceEvent) -> bool: ...

    def add_diagnostic(self, trace_id: str, diagnostic: str) -> None: ...

    def get_events(self, trace_id: str) -> Sequence[TraceEvent]: ...

    async def wait_for_quiet(
        self,
        trace_id: str,
        *,
        timeout_seconds: float,
        quiet_period_seconds: float,
    ) -> None: ...

    def build_record(self, trace_id: str) -> TraceRecord: ...


class InMemoryTraceStore:
    """A bounded trace store safe for OTel processor and receiver threads."""

    def __init__(self, *, max_spans_per_trace: int = 2_000) -> None:
        if max_spans_per_trace <= 0:
            raise ValueError("max_spans_per_trace must be positive")
        self._max_spans = max_spans_per_trace
        self._events: dict[str, dict[str, TraceEvent]] = {}
        self._diagnostics: dict[str, list[str]] = {}
        self._dropped: dict[str, int] = {}
        self._updated: dict[str, float] = {}
        self._lock = threading.RLock()

    def _touch(self, trace_id: str) -> None:
        self._updated[trace_id] = time.monotonic()

    def add(self, event: TraceEvent) -> bool:
        with self._lock:
            trace = self._events.setdefault(event.trace_id, {})
            if event.span_id not in trace and len(trace) >= self._max_spans:
                self._dropped[event.trace_id] = self._dropped.get(event.trace_id, 0) + 1
                self._append_diagnostic(event.trace_id, "span limit reached")
                self._touch(event.trace_id)
                return False
            trace[event.span_id] = event
            self._touch(event.trace_id)
            return True

    def _append_diagnostic(self, trace_id: str, diagnostic: str) -> None:
        values = self._diagnostics.setdefault(trace_id, [])
        if diagnostic not in values:
            values.append(diagnostic)

    def add_diagnostic(self, trace_id: str, diagnostic: str) -> None:
        with self._lock:
            self._append_diagnostic(trace_id, diagnostic)
            self._touch(trace_id)

    def get_events(self, trace_id: str) -> Sequence[TraceEvent]:
        with self._lock:
            values = list(self._events.get(trace_id, {}).values())
        return tuple(sorted(values, key=lambda event: (event.started_at, event.span_id)))

    async def wait_for_quiet(
        self,
        trace_id: str,
        *,
        timeout_seconds: float,
        quiet_period_seconds: float,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            with self._lock:
                updated = self._updated.get(trace_id)
                trace = self._events.get(trace_id, {})
                has_internal = any(
                    event.attributes.get(_RUNNER_ROOT_ATTRIBUTE) is not True
                    for event in trace.values()
                )
            if (
                updated is not None
                and has_internal
                and loop.time() - updated >= quiet_period_seconds
            ):
                return
            await asyncio.sleep(min(quiet_period_seconds / 2, 0.02))

    def build_record(self, trace_id: str) -> TraceRecord:
        events = list(self.get_events(trace_id))
        with self._lock:
            diagnostics = list(self._diagnostics.get(trace_id, []))
            dropped = self._dropped.get(trace_id, 0)
        runner_roots = [
            event for event in events if event.attributes.get(_RUNNER_ROOT_ATTRIBUTE) is True
        ]
        root = runner_roots[0] if len(runner_roots) == 1 else None
        internal = [event for event in events if event not in runner_roots]
        span_ids = {event.span_id for event in events}
        orphans = [
            event
            for event in internal
            if event.parent_span_id is not None and event.parent_span_id not in span_ids
        ]
        if not internal:
            status: Literal["complete", "partial", "missing"] = "missing"
            root_span_id = None
            diagnostics.append("no target-internal spans were observed")
        elif root is None or orphans or dropped or diagnostics:
            status = "partial"
            root_span_id = None if root is None else root.span_id
            if root is None:
                diagnostics.append("runner root span is missing or ambiguous")
            if orphans:
                diagnostics.append(f"{len(orphans)} span(s) have missing parents")
        else:
            status = "complete"
            root_span_id = root.span_id
        return TraceRecord(
            trace_id=trace_id,
            status=status,
            root_span_id=root_span_id,
            events=events,
            diagnostics=list(dict.fromkeys(diagnostics)),
            dropped_spans=dropped,
        )


class JsonlTraceStore(InMemoryTraceStore):
    """An in-memory store with a flushed, redacted JSONL journal."""

    def __init__(self, path: str | Path, *, max_spans_per_trace: int = 2_000) -> None:
        super().__init__(max_spans_per_trace=max_spans_per_trace)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, event: TraceEvent) -> bool:
        if not super().add(event):
            return False
        record = {
            "received_at": to_utc_iso8601(utc_now()),
            "event": event.model_dump(mode="json"),
        }
        payload = canonical_json_bytes(redact_secrets(record)) + b"\n"
        with self._lock, self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return True
