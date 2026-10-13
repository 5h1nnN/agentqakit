"""Shared deterministic evaluator helpers."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import JsonValue

from agentqa.models import EvaluationResult, Violation
from agentqa.storage import redact_secrets
from agentqa.targets.utils import extract_path
from agentqa.traceview import ToolCall, TraceView


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


__all__ = [
    "ToolCall",
    "TraceView",
    "evaluator_result",
    "is_content_summary",
    "output_text",
    "path_value",
]
