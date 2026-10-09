"""Deterministic MutationSpec matching."""

import hashlib
import json
from typing import Any

from agentqa.models import InterceptionPoint, MutationMatch, MutationSpec
from agentqa.mutations.context import MutationTrialSession
from agentqa.mutations.utils import get_path, path_exists, safe_jsonable


def _probability_sample(
    trial_seed: int,
    mutation_id: str,
    interception: InterceptionPoint,
    call_index: int,
) -> float:
    material = (
        f"{trial_seed}\0{mutation_id}\0{interception.kind}\0{interception.name}\0{call_index}"
    ).encode()
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value / 2**64


def match_mutation(
    spec: MutationSpec,
    interception: InterceptionPoint,
    arguments: dict[str, Any],
    call_index: int,
    session: MutationTrialSession,
) -> MutationMatch:
    """Evaluate all conditions without consuming shared random state."""
    mutation_id = spec.metadata.name
    conditions: dict[str, bool] = {
        "target.kind": spec.target.kind == interception.kind,
        "target.name": spec.target.name == interception.name,
    }
    if spec.when.call_index is not None:
        conditions["when.callIndex"] = spec.when.call_index == call_index
    if spec.when.mode == "first":
        conditions["when.mode.first"] = mutation_id not in session.applied_mutations

    normalized_arguments = safe_jsonable(arguments)
    for path, expected in spec.when.arguments.items():
        actual = get_path(normalized_arguments, path)
        conditions[f"when.arguments.{path}"] = actual == expected
    for path in spec.when.argument_exists:
        conditions[f"when.argumentExists.{path}"] = path_exists(normalized_arguments, path)
    for key, expected in spec.when.test_labels.items():
        conditions[f"when.testLabels.{key}"] = session.run_context.test_labels.get(key) == expected

    sample = _probability_sample(
        session.run_context.trial_seed or session.run_context.random_seed,
        mutation_id,
        interception,
        call_index,
    )
    conditions["when.probability"] = sample < spec.when.probability
    matched = all(conditions.values())
    failed = [name for name, value in conditions.items() if not value]
    reason = None if matched else f"condition did not match: {failed[0]}"
    return MutationMatch(
        mutation_id=mutation_id,
        matched=matched,
        conditions=conditions,
        reason=reason,
        probability_sample=sample,
    )


def canonical_arguments(arguments: dict[str, Any]) -> str:
    """Stable debug helper used by matcher tests without exposing raw values."""
    return json.dumps(safe_jsonable(arguments), sort_keys=True, separators=(",", ":"))
