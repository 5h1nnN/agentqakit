"""JSONL subprocess worker for Python callable targets."""

import asyncio
import contextlib
import importlib
import inspect
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from agentqa.mutations import (
    MutationEngine,
    MutationTrialSession,
    reset_current_mutation_session,
    set_current_mutation_session,
)
from agentqa.replay import (
    InteractionEngine,
    JsonlCassetteStore,
    ReplaySession,
    reset_current_replay_session,
    set_current_replay_session,
)
from agentqa.sdk import _reset_interaction_engine, _set_interaction_engine
from agentqa.targets.python_sdk import PythonSDKTrialConfig, PythonSDKTrialEvidence
from agentqa.tracing import InMemoryTraceStore
from agentqa.tracing.context import (
    reset_current_trace_store,
    set_current_trace_store,
)


def _safe_stack(exc: BaseException) -> list[dict[str, str | int]]:
    return [
        {
            "file": Path(frame.filename).name,
            "line": frame.lineno or 0,
            "function": frame.name,
        }
        for frame in traceback.extract_tb(exc.__traceback__)[-20:]
    ]


def _load(entrypoint: str, log: TextIO) -> Any:
    module_name, attribute = entrypoint.split(":", 1)
    with contextlib.redirect_stdout(log):
        module = importlib.import_module(module_name)
    target = getattr(module, attribute)
    if not callable(target):
        raise TypeError("configured Python entrypoint is not callable")
    return target


async def _call(target: Any, payload: dict[str, Any], log: TextIO) -> Any:
    with contextlib.redirect_stdout(log):
        result = target(payload)
        if inspect.isawaitable(result):
            result = await result
    return result


@dataclass(slots=True)
class _SDKState:
    config: PythonSDKTrialConfig
    mutation_session: MutationTrialSession
    replay_session: ReplaySession
    trace_store: InMemoryTraceStore
    trace_token: Any
    mutation_token: Any
    replay_token: Any
    engine_token: Any

    def evidence(self) -> PythonSDKTrialEvidence:
        self.replay_session.store.flush()
        trace_id = self.config.context.trace_id or ""
        return PythonSDKTrialEvidence(
            mutation_applications=self.mutation_session.applications,
            trace_events=list(self.trace_store.get_events(trace_id)),
            replay=self.replay_session.stats(),
            cassette_entries=list(self.replay_session.store.recorded_entries),
        )

    def close(self) -> None:
        _reset_interaction_engine(self.engine_token)
        reset_current_replay_session(self.replay_token)
        reset_current_mutation_session(self.mutation_token)
        reset_current_trace_store(self.trace_token)


def _activate_sdk(raw: Any) -> _SDKState:
    config = PythonSDKTrialConfig.model_validate_json(json.dumps(raw, ensure_ascii=False))
    context = config.context
    specs = {spec.metadata.name: spec for spec in config.mutations}
    mutation_session = MutationTrialSession(context)
    cassette_store = JsonlCassetteStore(
        source_path=None if config.replay.source_path is None else Path(config.replay.source_path),
        output_path=None if config.replay.output_path is None else Path(config.replay.output_path),
        max_inline_result_bytes=config.replay.max_inline_result_bytes,
    )
    replay_session = ReplaySession(
        run_context=context,
        mode=config.replay.mode,
        store=cassette_store,
        allow_live_fallback=config.replay.allow_live_fallback,
        volatile_fields=tuple(config.replay.volatile_fields),
    )
    trace_store = InMemoryTraceStore(
        max_spans_per_trace=config.trace.max_spans_per_trace,
    )
    interaction_engine = InteractionEngine(MutationEngine(specs))
    trace_token = set_current_trace_store(trace_store)
    mutation_token = set_current_mutation_session(mutation_session)
    replay_token = set_current_replay_session(replay_session)
    engine_token = _set_interaction_engine(interaction_engine)
    return _SDKState(
        config=config,
        mutation_session=mutation_session,
        replay_session=replay_session,
        trace_store=trace_store,
        trace_token=trace_token,
        mutation_token=mutation_token,
        replay_token=replay_token,
        engine_token=engine_token,
    )


async def _handle(target: Any, request: dict[str, Any], log: TextIO) -> dict[str, Any]:
    request_id = request.get("request_id")
    payload = request.get("input")
    if not isinstance(request_id, str) or not request_id:
        return {
            "request_id": request_id,
            "ok": False,
            "error": {
                "code": "PYTHON_PROTOCOL_ERROR",
                "type": "ValueError",
                "message": "invalid request_id",
            },
        }
    if not isinstance(payload, dict):
        return {
            "request_id": request_id,
            "ok": False,
            "error": {
                "code": "PYTHON_PROTOCOL_ERROR",
                "type": "TypeError",
                "message": "input must be an object",
            },
        }
    sdk_state: _SDKState | None = None
    response: dict[str, Any]
    if "agentqa" in request:
        try:
            sdk_state = _activate_sdk(request["agentqa"])
        except Exception as exc:
            return {
                "request_id": request_id,
                "ok": False,
                "error": {
                    "code": "PYTHON_PROTOCOL_ERROR",
                    "type": type(exc).__name__,
                    "message": "worker received invalid AgentQAKit SDK configuration",
                },
            }
    try:
        output = await _call(target, payload, log)
    except BaseException as exc:
        response = {
            "request_id": request_id,
            "ok": False,
            "error": {
                "code": "PYTHON_EXCEPTION",
                "type": type(exc).__name__,
                "message": str(exc)[:2_000] or type(exc).__name__,
                "stack": _safe_stack(exc),
            },
        }
    else:
        try:
            json.dumps(output, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            response = {
                "request_id": request_id,
                "ok": False,
                "error": {
                    "code": "PYTHON_SERIALIZATION_ERROR",
                    "type": type(exc).__name__,
                    "message": "Python target returned a non-JSON-serializable value",
                },
            }
        else:
            response = {"request_id": request_id, "ok": True, "output": output}
    if sdk_state is not None:
        try:
            response["agentqa"] = sdk_state.evidence().model_dump(mode="json")
        except BaseException as exc:
            response = {
                "request_id": request_id,
                "ok": False,
                "error": {
                    "code": "PYTHON_PROTOCOL_ERROR",
                    "type": type(exc).__name__,
                    "message": "worker could not serialize AgentQAKit SDK evidence",
                    "stack": _safe_stack(exc),
                },
            }
        finally:
            sdk_state.close()
    return response


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m agentqa.targets.python_worker module:function", file=sys.stderr)
        return 2
    protocol_output = sys.stdout
    try:
        target = _load(sys.argv[1], sys.stderr)
    except BaseException as exc:
        response = {
            "request_id": None,
            "ok": False,
            "error": {
                "code": "PYTHON_WORKER_ERROR",
                "type": type(exc).__name__,
                "message": "worker could not load the configured entrypoint",
                "stack": _safe_stack(exc),
            },
        }
        print(json.dumps(response, ensure_ascii=False), file=protocol_output, flush=True)
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError
        except (json.JSONDecodeError, TypeError):
            response = {
                "request_id": None,
                "ok": False,
                "error": {
                    "code": "PYTHON_PROTOCOL_ERROR",
                    "type": "ValueError",
                    "message": "worker received invalid JSONL",
                },
            }
        else:
            response = asyncio.run(_handle(target, request, sys.stderr))
        print(json.dumps(response, ensure_ascii=False), file=protocol_output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
