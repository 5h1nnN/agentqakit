"""Deterministic, bounded, and redacted evidence selection for AI operations."""

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

from pydantic import JsonValue

from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    EvaluationResult,
    EvidenceBundle,
    EvidenceRecord,
    EvidenceReference,
    TargetResult,
    TraceRecord,
)
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.storage import redact_secrets, redact_text_secrets
from agentqa.traceview import TraceView

_MAX_ITEM_BYTES = 8 * 1024
_MAX_BUNDLE_BYTES = 256 * 1024


def _named_evidence_id(prefix: str, name: str) -> str:
    """Keep manifest-derived evidence IDs valid even for maximal resource names."""
    candidate = f"{prefix}-{name}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    available = 128 - len(prefix) - len(digest) - 2
    return f"{prefix}-{name[:available]}-{digest}"


def _safe_value(value: Any) -> JsonValue:
    redacted = cast(JsonValue, redact_secrets(value))
    if isinstance(redacted, str):
        return redact_text_secrets(redacted)
    if isinstance(redacted, list):
        return [_safe_value(item) for item in redacted]
    if isinstance(redacted, dict):
        return {str(key): _safe_value(item) for key, item in redacted.items()}
    return redacted


def _bounded_data(value: Any, *, max_bytes: int = _MAX_ITEM_BYTES) -> tuple[JsonValue, bool]:
    safe = _safe_value(value)
    payload = canonical_json_bytes(safe)
    if len(payload) <= max_bytes:
        return safe, False
    preview = payload[: max_bytes // 2].decode("utf-8", errors="replace")
    return {
        "sha256": sha256_digest(payload),
        "length_bytes": len(payload),
        "preview": preview,
        "truncated": True,
    }, True


def _record(
    evidence_id: str,
    kind: str,
    source_ref: str,
    summary: str,
    value: Any,
    *,
    max_bytes: int = _MAX_ITEM_BYTES,
) -> EvidenceRecord:
    safe_value = _safe_value(value)
    content_hash = sha256_digest(canonical_json_bytes(safe_value))
    data, truncated = _bounded_data(safe_value, max_bytes=max_bytes)
    return EvidenceRecord.model_validate(
        {
            "evidence_id": evidence_id,
            "kind": kind,
            "source_ref": redact_text_secrets(source_ref[:1_000]),
            "content_hash": content_hash,
            "summary": redact_text_secrets(summary[:1_000]),
            "truncated": truncated,
            "data": data,
        }
    )


def _tool_names(test_spec: AgentTestSpec, view: TraceView) -> set[str]:
    names = {call.name for call in view.tool_calls}
    expectation = test_spec.expect.tools
    if expectation is None:
        return names
    names.update(expectation.required)
    names.update(expectation.forbidden)
    names.update(expectation.argument_schemas)
    if isinstance(expectation.min_calls, dict):
        names.update(expectation.min_calls)
    if isinstance(expectation.max_calls, dict):
        names.update(expectation.max_calls)
    for order in expectation.order:
        names.update((order.before, order.after))
    return names


def _merge_state(current: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        existing = current.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _merge_state(existing, value)
        else:
            current[key] = deepcopy(value)


def _policy_entries(manifest: AgentManifest) -> list[tuple[str, JsonValue]]:
    if manifest.policies is None:
        return []
    entries: list[tuple[str, JsonValue]] = [
        (
            item.id,
            {
                "id": item.id,
                "description": item.description,
                "severity": item.severity,
                "provenance": manifest.provenance.default_level,
            },
        )
        for item in manifest.policies.invariants
    ]
    entries.extend(
        (
            f"forbidden-behavior-{index}",
            {
                "id": f"forbidden-behavior-{index}",
                "forbidden_behavior": behavior,
                "provenance": manifest.provenance.default_level,
            },
        )
        for index, behavior in enumerate(manifest.policies.forbidden_behaviors, start=1)
    )
    entries.extend(
        (
            f"permission-requirement-{index}",
            {
                "id": f"permission-requirement-{index}",
                "permission_requirement": requirement,
                "provenance": manifest.provenance.default_level,
            },
        )
        for index, requirement in enumerate(manifest.policies.permission_requirements, start=1)
    )
    return entries


def _selected_names(
    available: Sequence[str],
    requested: Sequence[str],
    *,
    limit: int,
    label: str,
    missing: list[str],
) -> list[str]:
    available_set = set(available)
    if requested:
        if len(requested) != len(set(requested)):
            missing.append(f"duplicate requested manifest {label} selectors were ignored")
        unknown = sorted(set(requested) - available_set)
        missing.extend(f"requested manifest {label} {name!r} is unavailable" for name in unknown)
        selected = list(dict.fromkeys(name for name in requested if name in available_set))
    else:
        selected = list(available)
    if len(selected) > limit:
        missing.append(f"manifest {label} evidence was limited to {limit} entries")
    return selected[:limit]


def build_evidence_bundle(
    *,
    run_id: str,
    trial_id: str,
    test_spec: AgentTestSpec,
    target_result: TargetResult,
    trace_record: TraceRecord,
    manifest: AgentManifest | None = None,
    evaluations: Sequence[EvaluationResult] = (),
    requested_capabilities: Sequence[str] = (),
    requested_policies: Sequence[str] = (),
    max_scenario_bytes: int = _MAX_ITEM_BYTES,
    max_output_bytes: int = _MAX_ITEM_BYTES,
) -> EvidenceBundle:
    """Build stable selected evidence without sending full source or full traces."""
    items: list[EvidenceRecord] = []
    missing: list[str] = []
    blocking: list[str] = []
    view = TraceView(trace_record)

    items.append(
        _record(
            "scenario",
            "scenario",
            f"test:{test_spec.metadata.name}",
            "Declared test scenario and initial state",
            {
                "input": test_spec.scenario.input,
                "messages": [item.model_dump(mode="json") for item in test_spec.scenario.messages],
                "initial_state": test_spec.scenario.initial_state,
            },
            max_bytes=min(max_scenario_bytes, _MAX_ITEM_BYTES),
        )
    )
    items.append(
        _record(
            "target-result",
            "target",
            f"trial:{trial_id}",
            "Normalized target status, output, and safe error",
            {
                "status": target_result.status,
                "output": target_result.output,
                "error": (
                    None
                    if target_result.error is None
                    else target_result.error.model_dump(mode="json")
                ),
                "metadata": target_result.metadata,
            },
            max_bytes=min(max_output_bytes, _MAX_ITEM_BYTES),
        )
    )
    items.append(
        _record(
            "trace-completeness",
            "trace",
            f"trace:{trace_record.trace_id}",
            "Trace completeness, dropped spans, and collection diagnostics",
            {
                "status": trace_record.status,
                "dropped_spans": trace_record.dropped_spans,
                "diagnostics": trace_record.diagnostics[:20],
                "selected_event_count": len(view.internal_events),
                "full_trace_sent": False,
            },
        )
    )
    if trace_record.status != "complete":
        gap = f"trace is {trace_record.status}; trajectory evidence is incomplete"
        missing.append(gap)
        blocking.append(gap)

    if manifest is None:
        missing.append("AgentManifest was not supplied")
    else:
        if manifest.purpose is not None:
            items.append(
                _record(
                    "manifest-purpose",
                    "manifest",
                    "manifest:purpose",
                    "Declared agent purpose and out-of-scope behavior",
                    manifest.purpose.model_dump(mode="json"),
                )
            )
        capabilities = {item.name: item for item in manifest.capabilities}
        selected_capabilities = _selected_names(
            list(capabilities),
            requested_capabilities,
            limit=10,
            label="capability",
            missing=missing,
        )
        for name in selected_capabilities:
            item = capabilities[name]
            items.append(
                _record(
                    _named_evidence_id("manifest-capability", name),
                    "manifest",
                    f"manifest:capability/{name}",
                    f"Declared capability {name}",
                    {
                        **item.model_dump(mode="json"),
                        "provenance": manifest.provenance.default_level,
                    },
                )
            )
        policies = dict(_policy_entries(manifest))
        selected_policies = _selected_names(
            list(policies),
            requested_policies,
            limit=10,
            label="policy",
            missing=missing,
        )
        for name in selected_policies:
            items.append(
                _record(
                    _named_evidence_id("manifest-policy", name),
                    "manifest",
                    f"manifest:policy/{name}",
                    f"Declared policy {name}",
                    policies[name],
                )
            )
        tools = {item.name: item for item in manifest.tools}
        selected_tools = sorted(_tool_names(test_spec, view) & set(tools))
        if len(selected_tools) > 20:
            missing.append("manifest tool evidence was limited to 20 entries")
        for name in selected_tools[:20]:
            tool = tools[name]
            items.append(
                _record(
                    _named_evidence_id("manifest-tool", name),
                    "manifest",
                    f"manifest:tool/{name}",
                    f"Declared side-effect and confirmation semantics for tool {name}",
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "side_effect": tool.side_effect,
                        "risk": tool.risk,
                        "idempotent": tool.idempotent,
                        "requires_confirmation": tool.requires_confirmation,
                        "provenance": manifest.provenance.default_level,
                    },
                )
            )

    deterministic = [item for item in evaluations if item.evaluator_type != "llm_judge"]
    expected_deterministic = [
        item.type for item in test_spec.evaluators if item.type != "llm_judge"
    ]
    observed_types = {item.evaluator_type for item in deterministic}
    missing_types = [item for item in expected_deterministic if item not in observed_types]
    if missing_types:
        gap = "deterministic evaluator results are missing: " + ", ".join(missing_types)
        missing.append(gap)
        blocking.append(gap)
    for index, evaluation in enumerate(deterministic[:10], start=1):
        items.append(
            _record(
                f"evaluation-{index:02d}",
                "evaluation",
                f"evaluation:{evaluation.evaluator_type}",
                f"Deterministic {evaluation.evaluator_type} evaluator result",
                evaluation.model_dump(mode="json"),
            )
        )
    if len(deterministic) > 10:
        missing.append("deterministic evaluator evidence was limited to 10 results")

    state = deepcopy(test_spec.scenario.initial_state)
    state_events = [item for item in view.events if item.state_diff is not None]
    if test_spec.expect.state is not None and not state_events:
        gap = "required state transitions were not observed"
        missing.append(gap)
        blocking.append(gap)
    for index, event in enumerate(state_events[:10], start=1):
        before = deepcopy(state)
        assert event.state_diff is not None
        _merge_state(state, event.state_diff)
        items.append(
            _record(
                f"state-{index:02d}",
                "state",
                f"trace-span:{event.span_id}",
                f"State before/after selected span {event.name}",
                {
                    "span_id": event.span_id,
                    "span_name": event.name,
                    "before": before,
                    "diff": event.state_diff,
                    "after": state,
                },
            )
        )
    if len(state_events) > 10:
        missing.append("state transition evidence was limited to 10 events")

    for index, call in enumerate(view.tool_calls[:20], start=1):
        items.append(
            _record(
                f"tool-call-{index:02d}",
                "tool",
                f"trace-span:{call.event.span_id}",
                f"Selected arguments and result summary for tool {call.name}",
                {
                    "span_id": call.event.span_id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": call.event.output,
                    "status": call.event.status,
                },
            )
        )
    if len(view.tool_calls) > 20:
        missing.append("tool-call evidence was limited to 20 calls")

    errors = [item for item in view.internal_events if item.status == "ERROR"]
    for index, event in enumerate(errors[:8], start=1):
        items.append(
            _record(
                f"error-span-{index:02d}",
                "error",
                f"trace-span:{event.span_id}",
                f"Selected ERROR span {event.name}",
                {
                    "span_id": event.span_id,
                    "parent_span_id": event.parent_span_id,
                    "kind": event.kind,
                    "name": event.name,
                    "error": (None if event.error is None else event.error.model_dump(mode="json")),
                },
            )
        )
    if len(errors) > 8:
        missing.append("ERROR span evidence was limited to 8 events")

    bundle = EvidenceBundle(
        run_id=run_id,
        test_case_id=test_spec.metadata.name,
        trial_id=trial_id,
        trace_status=trace_record.status,
        items=items,
        missing_evidence=list(dict.fromkeys(missing)),
        blocking_gaps=list(dict.fromkeys(blocking)),
    )
    if len(canonical_json_bytes(bundle)) > _MAX_BUNDLE_BYTES:
        raise ValueError("selected AI evidence bundle exceeds 262144 bytes")
    return bundle


def evidence_bundle_hash(bundle: EvidenceBundle) -> str:
    """Hash the exact redacted evidence supplied to an AI operation."""
    return sha256_digest(canonical_json_bytes(bundle))


def evidence_index(bundle: EvidenceBundle) -> list[EvidenceReference]:
    """Strip item data while retaining stable traceability metadata."""
    return [
        EvidenceReference.model_validate(item.model_dump(mode="json", exclude={"data"}))
        for item in bundle.items
    ]


__all__ = ["build_evidence_bundle", "evidence_bundle_hash", "evidence_index"]
