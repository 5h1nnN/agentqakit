"""Small loopback-only OTLP/HTTP receiver for local AgentQA runs."""

import gzip
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from agentqa.tracing.options import TraceOptions
from agentqa.tracing.otlp import OTLPDecodeError, ingest_otlp_protobuf
from agentqa.tracing.store import TraceStore

OTLP_TRACES_PATH = "/v1/traces"


def _handler_factory(
    store: TraceStore,
    options: TraceOptions,
) -> Callable[..., BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status.value)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != OTLP_TRACES_PATH:
                self._reply(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain")
                return
            content_type = self.headers.get_content_type()
            if content_type != "application/x-protobuf":
                self._reply(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    b"expected application/x-protobuf\n",
                    "text/plain",
                )
                return
            try:
                content_length = int(self.headers.get("content-length", ""))
            except ValueError:
                content_length = -1
            if content_length < 0:
                self._reply(HTTPStatus.LENGTH_REQUIRED, b"content-length required\n", "text/plain")
                return
            if content_length > options.max_receiver_bytes:
                self.close_connection = True
                self._reply(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    b"payload too large\n",
                    "text/plain",
                )
                return
            payload = self.rfile.read(content_length)
            if self.headers.get("content-encoding", "").lower() == "gzip":
                try:
                    payload = gzip.decompress(payload)
                except (gzip.BadGzipFile, OSError):
                    self._reply(HTTPStatus.BAD_REQUEST, b"invalid gzip payload\n", "text/plain")
                    return
                if len(payload) > options.max_receiver_bytes:
                    self._reply(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        b"payload too large\n",
                        "text/plain",
                    )
                    return
            try:
                response = ingest_otlp_protobuf(payload, store, options)
            except OTLPDecodeError:
                self._reply(HTTPStatus.BAD_REQUEST, b"malformed protobuf\n", "text/plain")
                return
            except Exception:
                self._reply(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    b"trace ingestion failed\n",
                    "text/plain",
                )
                return
            self._reply(HTTPStatus.OK, response, "application/x-protobuf")

        def do_GET(self) -> None:
            self._reply(HTTPStatus.METHOD_NOT_ALLOWED, b"method not allowed\n", "text/plain")

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return Handler


class LocalOTLPReceiver:
    """Manage a background ThreadingHTTPServer bound to a loopback address."""

    def __init__(
        self,
        store: TraceStore,
        *,
        host: str = "127.0.0.1",
        port: int = 4318,
        options: TraceOptions | None = None,
    ) -> None:
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("the local OTLP receiver may only bind to loopback")
        if not 0 <= port <= 65_535:
            raise ValueError("port must be between 0 and 65535")
        self.options = options or TraceOptions()
        self._server = ThreadingHTTPServer((host, port), _handler_factory(store, self.options))
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        host = cast(str, host)
        display_host = f"[{host}]" if ":" in host else host
        return f"http://{display_host}:{port}"

    def start(self) -> None:
        """Start serving in a daemon thread."""
        if self._thread is not None:
            raise RuntimeError("OTLP receiver is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="agentqa-otlp-receiver",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the server and release its listening socket."""
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> "LocalOTLPReceiver":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.stop()
