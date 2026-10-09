"""JSONL subprocess worker for Python callable targets."""

import asyncio
import contextlib
import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any, TextIO


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
    try:
        output = await _call(target, payload, log)
        try:
            json.dumps(output, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            return {
                "request_id": request_id,
                "ok": False,
                "error": {
                    "code": "PYTHON_SERIALIZATION_ERROR",
                    "type": type(exc).__name__,
                    "message": "Python target returned a non-JSON-serializable value",
                },
            }
        return {"request_id": request_id, "ok": True, "output": output}
    except BaseException as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "error": {
                "code": "PYTHON_EXCEPTION",
                "type": type(exc).__name__,
                "message": str(exc)[:2_000] or type(exc).__name__,
                "stack": _safe_stack(exc),
            },
        }


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
