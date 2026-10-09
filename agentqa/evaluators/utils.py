"""Shared deterministic evaluator helpers."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from agentqa.models import EvaluationResult, TraceEvent, TraceRecord, Violation
from agentqa.storage import redact_secrets
from agentqa.targets.utils import extract_path


def output_text(value: JsonValue, *, max_chars: int) -> tuple[str, bool]:
    """Return stable bounded text plus whether the original value was truncated."""
    text = (
        value
        if isinstance(value, str)
        else json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return text[:max_chars], len(text) > max_chars


def path_value(value: Any, path: str) -> tuple[Any, bool]:
    """Resolve one restricted field path without executing expressions."""
    try:
        return extract_path(value, path), True
    except (KeyError, ValueError, TypeError):
        return None, False


def is_content_summary(value: Any) -> bool:
    """Identify T4/T5 non-reversible content summaries."""
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("sha256"), str)
        and ("length_bytes" in value or "bytes" in value)
        and set(value).issubset({"type", "length_bytes", "bytes", "sha256", "truncated", "preview"})
        and "preview" not in value
    )


def evaluator_result(
    evaluator_type: str,
    *,
    violations: Sequence[Violation] = (),
    unobservable: Sequence[str] = (),
    metadata: Mapping[str, JsonValue] | None = None,
    message: str | None = None,
) -> EvaluationResult:
    """Apply common FAIL-before-NOT_OBSERVABLE result precedence."""
    safe_metadata = redact_secrets(dict(metadata or {}))
    if not isinstance(safe_metadata, dict):
        safe_metadata = {}
    if violations:
        return EvaluationResult(
            evaluator_type=evaluator_type,
            status="FAIL",
            score=0.0,
            message=message or f"{len(violations)} assertion(s) failed",
            violations=list(violations),
            metadata=safe_metadata,
        )
    if unobservable:
        return EvaluationResult(
            evaluator_type=evaluator_type,
            status="NOT_OBSERVABLE",
            message=message or "; ".join(unobservable),
            metadata={**safe_metadata, "unobservable": list(unobservable)},
        )
    return EvaluationResult(
        evaluator_type=evaluator_type,
        status="PASS",
        score=1.0,
        message=message,
        metadata=safe_metadata,
    )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One logical tool call normalized from a framework or mutation event."""

    name: str
    event: TraceEvent
    arguments: JsonValue


def _logical_call_name(event: TraceEvent) -> str:
    interception_name = event.attributes.get("agentqa.interception.name")
    if isinstance(interception_name, str) and interception_name:
        return interception_name
    for key in ("tool.name", "gen_ai.tool.name", "tool_name"):
        value = event.attributes.get(key)
        if isinstance(value, str) and value:
            return value
    for prefix in (
        "agentqa.mutation.tool.",
        "agentqa.mutation.retriever.",
        "agentqa.replay.tool.",
        "agentqa.replay.retriever.",
    ):
        if event.name.startswith(prefix):
            return event.name.removeprefix(prefix)
    return event.name


def _logical_interactions(events: Sequence[TraceEvent]) -> list[TraceEvent]:
    """Prefer framework spans, then replay spans, over outer mutation spans."""
    ordinary_names = {
        _logical_call_name(event)
        for event in events
        if not event.name.startswith(("agentqa.mutation.", "agentqa.replay."))
    }
    replay_names = {
        _logical_call_name(event) for event in events if event.name.startswith("agentqa.replay.")
    }
    return [
        event
        for event in events
        if (
            not event.name.startswith(("agentqa.mutation.", "agentqa.replay."))
            or (
                event.name.startswith("agentqa.replay.")
                and _logical_call_name(event) not in ordinary_names
            )
            or (
                event.name.startswith("agentqa.mutation.")
                and _logical_call_name(event) not in ordinary_names
                and _logical_call_name(event) not in replay_names
            )
        )
    ]


class TraceView:
    """Chronological and parent-indexed view of normalized trace evidence."""

    def __init__(self, record: TraceRecord) -> None:
        self.status = record.status
        self.events = sorted(record.events, key=lambda item: (item.started_at, item.span_id))
        self.by_span_id = {event.span_id: event for event in self.events}
        self.children: dict[str, list[TraceEvent]] = {}
        for event in self.events:
            if event.parent_span_id is not None:
                self.children.setdefault(event.parent_span_id, []).append(event)

        raw_tools = _logical_interactions([event for event in self.events if event.kind == "TOOL"])
        self.tool_calls = [
            ToolCall(_logical_call_name(event), event, event.input) for event in raw_tools
        ]

    @property
    def internal_events(self) -> list[TraceEvent]:
        return [
            event
            for event in self.events
            if event.kind != "EVALUATOR" and event.attributes.get("agentqa.runner.root") is not True
        ]

    @property
    def step_count(self) -> int:
        non_call_events = [
            event for event in self.internal_events if event.kind not in {"TOOL", "RETRIEVER"}
        ]
        retriever_events = _logical_interactions(
            [event for event in self.internal_events if event.kind == "RETRIEVER"]
        )
        return len(non_call_events) + len(self.tool_calls) + len(retriever_events)

    @property
    def llm_calls(self) -> list[TraceEvent]:
        return [event for event in self.internal_events if event.kind == "LLM"]
