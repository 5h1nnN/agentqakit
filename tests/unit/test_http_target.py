from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentqa.models import HttpTargetSpec, RunContext
from agentqa.models import TestInput as TargetInput
from agentqa.targets import HttpTargetAdapter
from agentqa.targets.errors import TargetErrorCode, TargetSafetyError


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


def _spec(**updates: Any) -> HttpTargetSpec:
    data: dict[str, Any] = {
        "metadata": {"name": "http-fixture"},
        "type": "http",
        "url": "http://localhost:8765/chat",
        "request": {
            "headers": {"x-prompt": "{{ input.prompt }}"},
            "query": {"case": "{{ input.case }}"},
            "json": {"message": "{{ input.prompt }}"},
        },
        "response": {
            "outputPath": "$.result.answer",
            "allowedHeaders": ["authorization", "x-request-id"],
        },
    }
    data.update(updates)
    return HttpTargetSpec.model_validate(data)


async def _invoke(
    tmp_path: Path,
    response: httpx.Response | Exception,
    **spec_updates: Any,
) -> tuple[Any, httpx.Request | None]:
    captured: httpx.Request | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured
        captured = request
        if isinstance(response, Exception):
            raise response
        return response

    adapter = HttpTargetAdapter(
        _spec(**spec_updates),
        transport=httpx.MockTransport(handler),
    )
    context = _context(tmp_path)
    await adapter.prepare(context)
    result = await adapter.invoke(
        TargetInput(
            test_case_id="case-1",
            input={"prompt": "hello", "case": "one"},
        ),
        context,
    )
    await adapter.cleanup(context)
    return result, captured


async def test_http_adapter_renders_request_and_injects_trace_headers(tmp_path: Path) -> None:
    response = httpx.Response(
        200,
        json={"result": {"answer": "done"}},
        headers={"authorization": "secret", "x-request-id": "request-1"},
    )

    result, request = await _invoke(tmp_path, response)

    assert result.status == "OK"
    assert result.output == "done"
    assert request is not None
    assert request.url.params["case"] == "one"
    assert request.headers["x-prompt"] == "hello"
    assert request.headers["x-agentqa-run-id"] == "run-1"
    assert request.headers["traceparent"].startswith("00-12345678123412341234123456789abc-")
    assert result.metadata["headers"] == {
        "authorization": "***",
        "x-request-id": "request-1",
    }
    assert result.metadata["body"] == {"result": {"answer": "done"}}


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx.Response(500, json={"error": "failed"}), TargetErrorCode.HTTP_STATUS_ERROR),
        (httpx.Response(200, text="not-json"), TargetErrorCode.HTTP_INVALID_RESPONSE),
        (httpx.Response(200, content=b""), TargetErrorCode.HTTP_INVALID_RESPONSE),
    ],
)
async def test_http_adapter_normalizes_bad_responses(
    tmp_path: Path,
    response: httpx.Response,
    code: TargetErrorCode,
) -> None:
    result, _ = await _invoke(tmp_path, response)

    assert result.status == "ERROR"
    assert result.error is not None
    assert result.error.code == code.value


async def test_http_adapter_stops_oversized_response(tmp_path: Path) -> None:
    result, _ = await _invoke(
        tmp_path,
        httpx.Response(200, content=b'{"result":{"answer":"too large"}}'),
        maxResponseBytes=10,
    )

    assert result.status == "ERROR"
    assert result.error is not None
    assert result.error.code == TargetErrorCode.HTTP_RESPONSE_TOO_LARGE.value


async def test_http_adapter_normalizes_timeout_and_disconnect(tmp_path: Path) -> None:
    timeout, _ = await _invoke(tmp_path, httpx.ReadTimeout("slow"))
    disconnected, _ = await _invoke(tmp_path, httpx.ConnectError("gone"))

    assert timeout.status == "TIMEOUT"
    assert timeout.error is not None
    assert timeout.error.code == TargetErrorCode.HTTP_TIMEOUT.value
    assert disconnected.status == "ERROR"
    assert disconnected.error is not None
    assert disconnected.error.code == TargetErrorCode.HTTP_CONNECTION_ERROR.value


async def test_http_adapter_does_not_follow_redirects(tmp_path: Path) -> None:
    result, request = await _invoke(
        tmp_path,
        httpx.Response(302, headers={"location": "http://metadata.google.internal/"}),
    )

    assert request is not None
    assert request.url.host == "localhost"
    assert result.status == "ERROR"
    assert result.error is not None
    assert result.error.code == TargetErrorCode.HTTP_STATUS_ERROR.value


async def test_http_adapter_blocks_metadata_and_remote_by_default(tmp_path: Path) -> None:
    metadata = HttpTargetAdapter(_spec(url="http://169.254.169.254/latest"))
    with pytest.raises(TargetSafetyError, match="metadata"):
        await metadata.prepare(_context(tmp_path))

    remote = HttpTargetAdapter(_spec(url="http://8.8.8.8/chat"))
    with pytest.raises(TargetSafetyError, match="allowRemote"):
        await remote.prepare(_context(tmp_path))


async def test_http_adapter_does_not_trust_a_poisoned_localhost_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HttpTargetAdapter(_spec())

    async def resolve_to_public(hostname: str, port: int) -> set[Any]:
        del hostname, port
        import ipaddress

        return {ipaddress.ip_address("8.8.8.8")}

    monkeypatch.setattr(adapter, "_resolve", resolve_to_public)

    with pytest.raises(TargetSafetyError, match="allowRemote"):
        await adapter.prepare(_context(tmp_path))


async def test_http_adapter_rejects_transport_control_headers(tmp_path: Path) -> None:
    result, _ = await _invoke(
        tmp_path,
        httpx.Response(200, json={"result": {"answer": "unused"}}),
        request={"headers": {"host": "internal.example"}},
    )

    assert result.status == "ERROR"
    assert result.error is not None
    assert result.error.code == TargetErrorCode.CONFIGURATION_ERROR.value


def test_http_spec_requires_explicit_remote_before_private_network() -> None:
    with pytest.raises(ValueError, match="allowPrivateNetwork requires allowRemote"):
        _spec(allowPrivateNetwork=True)
