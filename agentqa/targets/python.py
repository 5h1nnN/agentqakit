"""Subprocess-isolated Python callable target adapter."""

import asyncio
import contextlib
import json
import os
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agentqa.conventions import new_id, utc_now
from agentqa.models import ErrorInfo, PythonTargetSpec, RunContext, TargetResult, TestInput
from agentqa.storage import redact_text_secrets
from agentqa.targets.errors import TargetConfigurationError, TargetErrorCode

_MAX_STDERR_BYTES = 65_536


@dataclass(slots=True)
class _WorkerProcess:
    process: asyncio.subprocess.Process
    stderr_task: asyncio.Task[None]
    stderr: bytearray = field(default_factory=bytearray)


class PythonTargetAdapter:
    """Execute one Python callable through a strict JSONL worker protocol."""

    def __init__(self, spec: PythonTargetSpec) -> None:
        self._spec = spec
        self._persistent: _WorkerProcess | None = None
        self._lock = asyncio.Lock()

    def _working_directory(self) -> Path:
        if self._spec.working_directory is None:
            return Path.cwd()
        directory = Path(self._spec.working_directory).expanduser().resolve()
        if not directory.is_dir():
            raise TargetConfigurationError("Python target workingDirectory is not a directory")
        return directory

    def _environment(self, context: RunContext) -> dict[str, str]:
        environment = {
            name: os.environ[name]
            for name in self._spec.environment_allowlist
            if name in os.environ
        }
        environment.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "AGENTQA_RUN_ID": context.run_id,
                "AGENTQA_TEST_ID": context.test_case_id or "",
                "AGENTQA_TRIAL_ID": context.trial_id or "",
                "AGENTQA_TRACE_ID": context.trace_id or "",
                "AGENTQA_MUTATION_ID": context.mutation_id or "",
                "AGENTQA_ARTIFACT_DIR": context.artifact_dir or "",
            }
        )
        if context.otlp_endpoint is not None:
            environment.update(
                {
                    "OTEL_EXPORTER_OTLP_ENDPOINT": context.otlp_endpoint,
                    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                    "OTEL_TRACES_EXPORTER": "otlp",
                    "OTEL_SERVICE_NAME": "agentqa-python-target",
                }
            )
        return environment

    @staticmethod
    async def _capture_stderr(stream: asyncio.StreamReader | None, output: bytearray) -> None:
        if stream is None:
            return
        while chunk := await stream.read(4096):
            remaining = _MAX_STDERR_BYTES - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])

    async def _start_worker(self, context: RunContext) -> _WorkerProcess:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "agentqa.targets.python_worker",
            self._spec.entrypoint,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._working_directory(),
            env=self._environment(context),
            start_new_session=True,
        )
        stderr = bytearray()
        task = asyncio.create_task(self._capture_stderr(process.stderr, stderr))
        return _WorkerProcess(process=process, stderr_task=task, stderr=stderr)

    @staticmethod
    async def _stop_worker(worker: _WorkerProcess, *, grace_seconds: float = 0.25) -> None:
        process = worker.process
        if process.stdin is not None:
            process.stdin.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.wait_closed()
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=grace_seconds)
            except TimeoutError:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)  # type: ignore[attr-defined]
                else:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=grace_seconds)
                except TimeoutError:
                    if os.name == "posix":
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(  # type: ignore[attr-defined]
                                process.pid, signal.SIGKILL  # type: ignore[attr-defined]
                            )
                    else:
                        process.kill()
                    await process.wait()
        await asyncio.gather(worker.stderr_task, return_exceptions=True)

    async def prepare(self, context: RunContext) -> None:
        self._working_directory()
        if self._spec.worker_mode == "persistent":
            self._persistent = await self._start_worker(context)

    async def cleanup(self, context: RunContext) -> None:
        del context
        if self._persistent is not None:
            await self._stop_worker(self._persistent)
            self._persistent = None

    @staticmethod
    def _stderr_metadata(worker: _WorkerProcess) -> dict[str, Any]:
        if not worker.stderr:
            return {}
        return {
            "worker_log": redact_text_secrets(worker.stderr.decode("utf-8", errors="replace")),
            "worker_log_truncated": len(worker.stderr) >= _MAX_STDERR_BYTES,
        }

    @staticmethod
    def _error_result(
        *,
        started_at: Any,
        code: str,
        message: str,
        status: Literal["ERROR", "TIMEOUT"] = "ERROR",
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TargetResult:
        return TargetResult(
            status=status,
            error=ErrorInfo(code=code, message=message, details=details or {}),
            started_at=started_at,
            ended_at=utc_now(),
            metadata=metadata or {},
        )

    async def _exchange(
        self,
        worker: _WorkerProcess,
        test_input: TestInput,
    ) -> TargetResult:
        started_at = utc_now()
        request_id = new_id()
        process = worker.process
        if process.stdin is None or process.stdout is None:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.PYTHON_WORKER_ERROR.value,
                message="Python worker streams are unavailable",
            )
        request = {
            "request_id": request_id,
            "input": test_input.model_dump(mode="json"),
        }
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False).encode() + b"\n")
            await process.stdin.drain()
            line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=self._spec.timeout_ms / 1000,
            )
        except TimeoutError:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.PYTHON_TIMEOUT.value,
                message="Python target timed out",
                status="TIMEOUT",
                metadata=self._stderr_metadata(worker),
            )
        except (BrokenPipeError, ConnectionResetError):
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.PYTHON_WORKER_ERROR.value,
                message="Python worker exited before sending a response",
                metadata=self._stderr_metadata(worker),
            )
        if not line:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.PYTHON_WORKER_ERROR.value,
                message="Python worker exited without a response",
                details={"return_code": process.returncode},
                metadata=self._stderr_metadata(worker),
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.PYTHON_PROTOCOL_ERROR.value,
                message="Python worker returned invalid JSONL",
                metadata=self._stderr_metadata(worker),
            )
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.PYTHON_PROTOCOL_ERROR.value,
                message="Python worker returned a mismatched request_id",
                metadata=self._stderr_metadata(worker),
            )
        if response.get("ok") is True:
            return TargetResult(
                status="OK",
                output=response.get("output"),
                started_at=started_at,
                ended_at=utc_now(),
                metadata={
                    **self._stderr_metadata(worker),
                    "latency_ms": (utc_now() - started_at).total_seconds() * 1000,
                    "worker_mode": self._spec.worker_mode,
                },
            )
        error = response.get("error")
        if not isinstance(error, dict):
            error = {}
        code = str(error.get("code") or TargetErrorCode.PYTHON_WORKER_ERROR.value)
        known_codes = {item.value for item in TargetErrorCode}
        if code not in known_codes:
            code = TargetErrorCode.PYTHON_WORKER_ERROR.value
        details: dict[str, Any] = {}
        if isinstance(error.get("type"), str):
            details["exception_type"] = error["type"]
        if isinstance(error.get("stack"), list):
            details["stack"] = error["stack"]
        message = error.get("message")
        if not isinstance(message, str) or not message:
            message = "Python worker reported an error"
        message = redact_text_secrets(message)
        return self._error_result(
            started_at=started_at,
            code=code,
            message=message,
            details=details,
            metadata=self._stderr_metadata(worker),
        )

    async def invoke(self, test_input: TestInput, context: RunContext) -> TargetResult:
        async with self._lock:
            worker = self._persistent
            owns_worker = worker is None
            if worker is None:
                try:
                    worker = await self._start_worker(context)
                except (OSError, TargetConfigurationError) as exc:
                    now = utc_now()
                    return self._error_result(
                        started_at=now,
                        code=TargetErrorCode.PYTHON_WORKER_ERROR.value,
                        message=f"could not start Python worker: {type(exc).__name__}",
                    )
            try:
                result = await self._exchange(worker, test_input)
                if not owns_worker and result.status == "TIMEOUT":
                    await self._stop_worker(worker)
                    self._persistent = None
                return result
            finally:
                if owns_worker:
                    await self._stop_worker(worker)
                elif worker.process.returncode is not None:
                    self._persistent = None
