"""Deterministic JSON Schema export for public data contracts."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, TypeAdapter

from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    EvaluationResult,
    GateResult,
    MutationSpec,
    RunContext,
    RunResult,
    TargetResult,
    TargetSpec,
    TestResult,
    TraceEvent,
)

_SCHEMA_MODELS: dict[str, type[BaseModel] | TypeAdapter[Any]] = {
    "agent-manifest": AgentManifest,
    "agent-test-spec": AgentTestSpec,
    "evaluation-result": EvaluationResult,
    "gate-result": GateResult,
    "mutation-spec": MutationSpec,
    "run-context": RunContext,
    "run-result": RunResult,
    "target-result": TargetResult,
    "target-spec": TypeAdapter(cast(Any, TargetSpec)),
    "test-result": TestResult,
    "trace-event": TraceEvent,
}


def build_json_schemas() -> dict[str, dict[str, Any]]:
    """Build all public JSON Schemas with their external field names."""
    schemas: dict[str, dict[str, Any]] = {}
    for name, model in _SCHEMA_MODELS.items():
        if isinstance(model, TypeAdapter):
            schemas[name] = model.json_schema(by_alias=True)
        else:
            schemas[name] = model.model_json_schema(by_alias=True)
    return schemas


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def export_json_schemas(directory: str | Path) -> list[Path]:
    """Write all schemas using stable formatting and atomic replacement."""
    output = Path(directory)
    written: list[Path] = []
    for name, schema in build_json_schemas().items():
        path = output / f"{name}.schema.json"
        payload = (json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        _atomic_write(path, payload)
        written.append(path)
    return written
