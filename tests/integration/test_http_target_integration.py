import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agentqa.models import HttpTargetSpec, RunContext
from agentqa.models import TestInput as TargetInput
from agentqa.targets import HttpTargetAdapter

pytestmark = pytest.mark.integration


class _JsonHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        response = json.dumps({"answer": f"received:{request['message']}"}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def _http_server() -> Iterator[str]:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _JsonHandler)
    except PermissionError:
        pytest.skip("the execution sandbox does not permit loopback socket binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        assert isinstance(host, str)
        yield f"http://{host}:{port}/chat"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def test_http_adapter_calls_real_loopback_service(tmp_path: Path) -> None:
    with _http_server() as url:
        spec = HttpTargetSpec.model_validate(
            {
                "metadata": {"name": "real-http"},
                "type": "http",
                "url": url,
                "request": {"json": {"message": "{{ input.message }}"}},
                "response": {"outputPath": "$.answer"},
            }
        )
        context = RunContext(
            run_id="run-1",
            spec_hash="spec-hash",
            started_at=datetime.now(UTC),
            random_seed=1,
            artifact_dir=str(tmp_path),
            test_case_id="real-http-case",
            trace_id="12345678-1234-1234-1234-123456789abc",
        )
        adapter = HttpTargetAdapter(spec)
        await adapter.prepare(context)

        result = await adapter.invoke(
            TargetInput(test_case_id="real-http-case", input={"message": "hello"}),
            context,
        )
        await adapter.cleanup(context)

    assert result.status == "OK"
    assert result.output == "received:hello"
