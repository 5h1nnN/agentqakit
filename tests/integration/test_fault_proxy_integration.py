import json
import threading
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentqa.models import MutationSpec
from agentqa.mutations import FaultProxy, FaultProxyError, FaultProxyOptions

pytestmark = pytest.mark.integration


class _UpstreamState:
    def __init__(self) -> None:
        self.calls = 0


@pytest.fixture
def upstream() -> Generator[tuple[_UpstreamState, str], None, None]:
    state = _UpstreamState()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state.calls += 1
            body = json.dumps(
                {
                    "answer": "ok",
                    "profile": {"name": "Ada", "email": "ada@example.test"},
                    "token": "upstream-secret",
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield state, f"http://{host!s}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _http_mutation(
    mutation_id: str,
    operation: str,
    **operation_values: Any,
) -> MutationSpec:
    return MutationSpec.model_validate(
        {
            "metadata": {"name": mutation_id},
            "target": {"kind": "http", "name": "campus.api"},
            "operation": {"type": operation, **operation_values},
        }
    )


def _request(endpoint: str, mutation_id: str | None = None) -> httpx.Response:
    headers = {
        "x-agentqa-test-id": "proxy-test",
        "x-agentqa-trial-id": f"trial-{mutation_id or 'baseline'}",
        "x-agentqa-trace-id": f"trace-{mutation_id or 'baseline'}",
        "x-agentqa-trial-seed": "7",
    }
    if mutation_id is not None:
        headers["x-agentqa-mutation-id"] = mutation_id
    return httpx.get(
        f"{endpoint}/tool",
        headers=headers,
        timeout=2,
        trust_env=False,
    )


def test_fault_proxy_rejects_open_proxy_configurations(tmp_path: Path) -> None:
    with pytest.raises(FaultProxyError, match="loopback"):
        FaultProxyOptions(
            upstream_base_url="https://example.test",
            allowed_paths=("/tool",),
            route_name="campus.api",
            cassette_directory=tmp_path,
            host="0.0.0.0",
        )
    with pytest.raises(FaultProxyError, match="at least one exact"):
        FaultProxyOptions(
            upstream_base_url="https://example.test",
            allowed_paths=(),
            route_name="campus.api",
            cassette_directory=tmp_path,
        )
    with pytest.raises(FaultProxyError, match="credentials"):
        FaultProxyOptions(
            upstream_base_url="https://user:password@example.test",
            allowed_paths=("/tool",),
            route_name="campus.api",
            cassette_directory=tmp_path,
        )


def test_proxy_records_sanitized_cassette_and_replays_without_upstream(
    tmp_path: Path,
    upstream: tuple[_UpstreamState, str],
) -> None:
    state, upstream_url = upstream
    cassette_directory = tmp_path / "cassettes"
    options = FaultProxyOptions(
        upstream_base_url=upstream_url,
        allowed_paths=("/tool",),
        route_name="campus.api",
        cassette_directory=cassette_directory,
    )
    proxy = FaultProxy(options)
    endpoint = proxy.start()
    try:
        response = _request(endpoint)
        forbidden = httpx.get(f"{endpoint}/not-allowed", timeout=2, trust_env=False)
    finally:
        proxy.stop()

    assert response.status_code == 200
    assert response.json()["token"] == "upstream-secret"
    assert response.headers["x-agentqa-proxy-source"] == "upstream"
    assert forbidden.status_code == 403
    assert state.calls == 1
    cassette_files = [
        path for path in cassette_directory.glob("*.json") if path.name != "applications.jsonl"
    ]
    assert len(cassette_files) == 1
    assert "upstream-secret" not in cassette_files[0].read_text(encoding="utf-8")

    replay = FaultProxy(
        FaultProxyOptions(
            upstream_base_url="http://127.0.0.1:1",
            allowed_paths=("/tool",),
            route_name="campus.api",
            cassette_directory=cassette_directory,
            mode="replay",
        )
    )
    replay_endpoint = replay.start()
    try:
        replayed = _request(replay_endpoint)
        miss = httpx.post(f"{replay_endpoint}/tool", timeout=2, trust_env=False)
    finally:
        replay.stop()

    assert replayed.status_code == 200
    assert replayed.json()["token"] == "***"
    assert replayed.headers["x-agentqa-proxy-source"] == "cassette"
    assert miss.status_code == 502
    assert state.calls == 1


def test_proxy_applies_http_mutations_selected_by_headers(
    tmp_path: Path,
    upstream: tuple[_UpstreamState, str],
) -> None:
    state, upstream_url = upstream
    mutations = {
        "delay": _http_mutation("delay", "timeout", delayMs=10),
        "status": _http_mutation("status", "http_status", statusCode=503),
        "invalid": _http_mutation("invalid", "invalid_json"),
        "missing": _http_mutation("missing", "missing_field", field="profile.email"),
        "wrong": _http_mutation(
            "wrong",
            "wrong_value",
            field="answer",
            value="mutated",
        ),
        "disconnect": _http_mutation("disconnect", "http_disconnect"),
    }
    proxy = FaultProxy(
        FaultProxyOptions(
            upstream_base_url=upstream_url,
            allowed_paths=("/tool",),
            route_name="campus.api",
            cassette_directory=tmp_path / "cassettes",
        ),
        mutations,
    )
    endpoint = proxy.start()
    try:
        started = time.monotonic()
        delayed = _request(endpoint, "delay")
        elapsed = time.monotonic() - started
        status = _request(endpoint, "status")
        invalid = _request(endpoint, "invalid")
        missing = _request(endpoint, "missing")
        wrong = _request(endpoint, "wrong")
        with pytest.raises((httpx.RemoteProtocolError, httpx.ReadError)):
            _request(endpoint, "disconnect")
    finally:
        proxy.stop()

    assert delayed.status_code == 200
    assert elapsed >= 0.01
    assert status.status_code == 503
    assert status.json()["error"] == "AGENTQA_INJECTED_HTTP_STATUS"
    assert invalid.text == "{agentqa-invalid-json"
    assert invalid.headers["x-agentqa-mutation-triggered"] == "true"
    assert "email" not in missing.json()["profile"]
    assert wrong.json()["answer"] == "mutated"
    assert state.calls == 4

    applications = (tmp_path / "cassettes" / "applications.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in applications.splitlines()]
    assert len(records) == 6
    assert all(record["triggered"] for record in records)
    assert {record["operation_type"] for record in records} == {
        "timeout",
        "http_status",
        "invalid_json",
        "missing_field",
        "wrong_value",
        "http_disconnect",
    }
