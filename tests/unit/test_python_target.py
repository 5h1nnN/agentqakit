from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentqa.models import PythonTargetSpec, RunContext
from agentqa.models import TestInput as TargetInput
from agentqa.targets import PythonTargetAdapter
from agentqa.targets.errors import TargetErrorCode


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        run_id="run-1",
        spec_hash="spec-hash",
        started_at=datetime.now(UTC),
        random_seed=7,
        artifact_dir=str(tmp_path),
        test_case_id="case-1",
        trial_id="trial-1",
        trace_id="12345678-1234-1234-1234-123456789abc",
    )


def _input(value: str = "hello") -> TargetInput:
    return TargetInput(test_case_id="case-1", input={"message": value})


def _spec(
    function: str,
    *,
    timeout_ms: int = 2_000,
    worker_mode: str = "perInvocation",
) -> PythonTargetSpec:
    return PythonTargetSpec.model_validate(
        {
            "metadata": {"name": "python-fixture"},
            "type": "python",
            "entrypoint": f"tests.fixtures.python_targets:{function}",
            "timeoutMs": timeout_ms,
            "workerMode": worker_mode,
        }
    )


@pytest.mark.parametrize("function", ["echo", "async_echo"])
async def test_python_adapter_runs_sync_and_async_callables(function: str, tmp_path: Path) -> None:
    adapter = PythonTargetAdapter(_spec(function))
    context = _context(tmp_path)
    await adapter.prepare(context)

    result = await adapter.invoke(_input(), context)

    await adapter.cleanup(context)
    assert result.status == "OK"
    assert result.output == {"received": {"message": "hello"}}


async def test_python_adapter_keeps_prints_out_of_protocol_and_redacts_log(
    tmp_path: Path,
) -> None:
    adapter = PythonTargetAdapter(_spec("prints"))
    context = _context(tmp_path)
    await adapter.prepare(context)

    result = await adapter.invoke(_input(), context)

    assert result.status == "OK"
    assert result.output == {"message": "hello"}
    assert result.metadata["worker_log"] == "worker user log Authorization: ***\n"


@pytest.mark.parametrize(
    ("function", "code"),
    [
        ("nonserializable", TargetErrorCode.PYTHON_SERIALIZATION_ERROR.value),
        ("raises", TargetErrorCode.PYTHON_EXCEPTION.value),
    ],
)
async def test_python_adapter_normalizes_worker_errors(
    function: str, code: str, tmp_path: Path
) -> None:
    adapter = PythonTargetAdapter(_spec(function))
    context = _context(tmp_path)

    result = await adapter.invoke(_input(), context)

    assert result.status == "ERROR"
    assert result.error is not None
    assert result.error.code == code
    if function == "raises":
        assert result.error.details["exception_type"] == "RuntimeError"
        assert isinstance(result.error.details["stack"], list)


async def test_python_adapter_times_out_and_terminates_exact_worker(tmp_path: Path) -> None:
    adapter = PythonTargetAdapter(_spec("waits_forever", timeout_ms=50))
    context = _context(tmp_path)

    result = await adapter.invoke(_input(), context)

    assert result.status == "TIMEOUT"
    assert result.error is not None
    assert result.error.code == TargetErrorCode.PYTHON_TIMEOUT.value


async def test_per_invocation_mode_prevents_global_state_crosstalk(tmp_path: Path) -> None:
    adapter = PythonTargetAdapter(_spec("counter"))
    context = _context(tmp_path)

    first = await adapter.invoke(_input("one"), context)
    second = await adapter.invoke(_input("two"), context)

    assert first.output == {"calls": 1}
    assert second.output == {"calls": 1}


async def test_persistent_mode_correlates_consecutive_requests(tmp_path: Path) -> None:
    adapter = PythonTargetAdapter(_spec("counter", worker_mode="persistent"))
    context = _context(tmp_path)
    await adapter.prepare(context)

    first = await adapter.invoke(_input("one"), context)
    second = await adapter.invoke(_input("two"), context)
    await adapter.cleanup(context)

    assert first.output == {"calls": 1}
    assert second.output == {"calls": 2}
