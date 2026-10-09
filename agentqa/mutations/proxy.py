"""Loopback-only fixed-upstream HTTP fault proxy with safe cassette replay."""

import base64
import json
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
from pydantic import JsonValue

from agentqa.conventions import utc_now
from agentqa.models import (
    InterceptionPoint,
    MutationApplication,
    MutationMatch,
    MutationSpec,
    RunContext,
)
from agentqa.mutations.context import MutationTrialSession
from agentqa.mutations.matcher import match_mutation
from agentqa.mutations.utils import delete_path, replace_path, safe_jsonable, summarize
from agentqa.replay.models import CassetteError
from agentqa.replay.store import JsonlCassetteStore, ReplayError
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.storage import redact_secrets
from agentqa.storage.redaction import redact_text_secrets

_CONTROL_HEADERS = {
    "x-agentqa-mutation-id",
    "x-agentqa-span-id",
    "x-agentqa-test-labels",
    "x-agentqa-trial-id",
    "x-agentqa-trial-seed",
}
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_CASSETTE_HEADER_DENYLIST = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
}


class FaultProxyError(ValueError):
    """A safe fixed-upstream proxy configuration error."""


@dataclass(frozen=True, slots=True)
class FaultProxyOptions:
    """Network and storage policy for one local fault proxy."""

    upstream_base_url: str
    allowed_paths: tuple[str, ...]
    route_name: str
    cassette_directory: Path
    mode: Literal["record", "replay"] = "record"
    host: str = "127.0.0.1"
    port: int = 0
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    max_request_bytes: int = 1_048_576
    max_response_bytes: int = 1_048_576
    allow_live_fallback: bool = False

    def __post_init__(self) -> None:
        parsed = urlsplit(self.upstream_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise FaultProxyError("upstream_base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise FaultProxyError("upstream_base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise FaultProxyError("upstream_base_url must not contain query or fragment")
        if self.host not in {"127.0.0.1", "::1", "localhost"}:
            raise FaultProxyError("fault proxy must bind to a loopback host")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.route_name):
            raise FaultProxyError("route_name must be a valid mutation resource name")
        if self.mode not in {"record", "replay"}:
            raise FaultProxyError("mode must be record or replay")
        if self.mode != "replay" and self.allow_live_fallback:
            raise FaultProxyError("allow_live_fallback is only valid in replay mode")
        if not 0 <= self.port <= 65_535:
            raise FaultProxyError("port must be between 0 and 65535")
        if not self.allowed_paths:
            raise FaultProxyError("at least one exact allowed path is required")
        for path in self.allowed_paths:
            if not path.startswith("/") or "?" in path or "#" in path:
                raise FaultProxyError("allowed paths must be absolute paths without query")
            if ".." in unquote(path).split("/"):
                raise FaultProxyError("allowed paths must not contain traversal segments")
        if self.connect_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise FaultProxyError("proxy timeouts must be positive")
        if self.max_request_bytes <= 0 or self.max_response_bytes <= 0:
            raise FaultProxyError("proxy body limits must be positive")


@dataclass(frozen=True, slots=True)
class _ProxyResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes
    source: Literal["upstream", "cassette", "synthetic"]


class _CassetteStore:
    def __init__(
        self,
        directory: Path,
        *,
        mode: Literal["record", "replay"],
        route_name: str,
        allow_live_fallback: bool,
    ) -> None:
        self._directory = directory.resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._route_name = route_name
        self.allow_live_fallback = allow_live_fallback
        self._call_index = 0
        source = self._directory / "cassette.jsonl" if mode == "replay" else None
        output = (
            self._directory / "cassette.jsonl"
            if mode == "record"
            else self._directory / "replay-delta.jsonl"
        )
        try:
            self._store = JsonlCassetteStore(
                source_path=source,
                output_path=output,
            )
        except ReplayError:
            raise FaultProxyError("cassette could not be loaded") from None

    def request(
        self,
        method: str,
        path_and_query: str,
        headers: dict[str, str],
        body: bytes,
    ) -> tuple[JsonValue, int]:
        with self._lock:
            self._call_index += 1
            call_index = self._call_index
        normalized_headers = {name.lower(): value for name, value in headers.items()}
        safe_body = self._safe_body(normalized_headers, body)
        return (
            {
                "method": method.upper(),
                "path_and_query": redact_text_secrets(path_and_query),
                "body_sha256": sha256_digest(safe_body),
            },
            call_index,
        )

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _safe_body(headers: dict[str, str], body: bytes) -> bytes:
        content_type = headers.get("content-type", "").lower()
        if "json" in content_type:
            try:
                parsed = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return b'{"agentqa_redacted":"invalid-json"}'
            return canonical_json_bytes(redact_secrets(parsed))
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return b"<agentqa-redacted-binary>"
        return redact_text_secrets(text).encode()

    def record(
        self,
        arguments: JsonValue,
        call_index: int,
        response: _ProxyResponse,
        *,
        trace_id: str,
        span_id: str,
    ) -> None:
        retained_headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower() not in _CASSETTE_HEADER_DENYLIST | _HOP_BY_HOP_HEADERS
        }
        redacted_headers = redact_secrets(retained_headers)
        headers = (
            {str(name): str(value) for name, value in redacted_headers.items()}
            if isinstance(redacted_headers, dict)
            else {}
        )
        normalized_headers = {name.lower(): value for name, value in headers.items()}
        safe_body = self._safe_body(normalized_headers, response.body)
        result = cast(
            JsonValue,
            {
                "status_code": response.status_code,
                "headers": headers,
                "body_base64": base64.b64encode(safe_body).decode("ascii"),
            },
        )
        self._store.record_success(
            kind="http",
            name=self._route_name,
            arguments=arguments,
            call_index=call_index,
            result=result,
            trace_id=trace_id,
            span_id=span_id,
        )
        self._store.flush()

    def replay(self, arguments: JsonValue, call_index: int) -> _ProxyResponse | None:
        entry = self._store.lookup("http", self._route_name, arguments, call_index)
        if entry is None:
            return None
        if entry.status == "error":
            raise FaultProxyError("replayed upstream error")
        try:
            payload = self._store.result(entry)
            if not isinstance(payload, dict):
                return None
            raw_headers = payload["headers"]
            if not isinstance(raw_headers, dict):
                return None
            headers = {str(name): str(value) for name, value in raw_headers.items()}
            return _ProxyResponse(
                status_code=int(cast(Any, payload["status_code"])),
                headers=headers,
                body=base64.b64decode(str(payload["body_base64"]), validate=True),
                source="cassette",
            )
        except (KeyError, OSError, TypeError, ValueError, ReplayError):
            return None

    def record_error(
        self,
        arguments: JsonValue,
        call_index: int,
        error: Exception,
        *,
        trace_id: str,
        span_id: str,
    ) -> None:
        self._store.record_error(
            kind="http",
            name=self._route_name,
            arguments=arguments,
            call_index=call_index,
            error=CassetteError(
                code="UPSTREAM_ERROR",
                exception_type=type(error).__name__,
                message=f"upstream raised {type(error).__name__}",
            ),
            trace_id=trace_id,
            span_id=span_id,
        )
        self._store.flush()

    def flush(self) -> None:
        self._store.flush()

    def append_application(self, application: MutationApplication) -> None:
        path = self._directory / "applications.jsonl"
        line = application.model_dump_json().encode() + b"\n"
        with self._lock:
            current = path.read_bytes() if path.exists() else b""
            payload = current + line
            self._atomic_write(path, payload)
            self._atomic_write(
                path.with_name(f"{path.name}.sha256"),
                f"{sha256_digest(payload)}\n".encode(),
            )


class FaultProxy:
    """A local reverse proxy that can mutate only one preconfigured upstream."""

    def __init__(
        self,
        options: FaultProxyOptions,
        mutations: dict[str, MutationSpec] | None = None,
    ) -> None:
        self.options = options
        self._mutations = dict(mutations or {})
        for name, spec in self._mutations.items():
            if name != spec.metadata.name:
                raise FaultProxyError("mutation mapping keys must equal metadata.name")
            if spec.target.kind != "http":
                raise FaultProxyError("fault proxy accepts only HTTP mutation specs")
        self._cassettes = _CassetteStore(
            options.cassette_directory,
            mode=options.mode,
            route_name=options.route_name,
            allow_live_fallback=options.allow_live_fallback,
        )
        self._sessions: dict[str, MutationTrialSession] = {}
        self._session_lock = threading.RLock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=options.connect_timeout_seconds,
                read=options.read_timeout_seconds,
                write=options.read_timeout_seconds,
                pool=options.connect_timeout_seconds,
            ),
            follow_redirects=False,
            trust_env=False,
        )

    @property
    def endpoint(self) -> str:
        if self._server is None:
            raise RuntimeError("fault proxy is not running")
        host, port = self._server.server_address[:2]
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else str(host)
        return f"http://{display_host}:{port}"

    def start(self) -> str:
        """Start the loopback server in a background thread."""
        if self._server is not None:
            raise RuntimeError("fault proxy is already running")
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                proxy._handle(self)

            def do_POST(self) -> None:
                proxy._handle(self)

            def do_PUT(self) -> None:
                proxy._handle(self)

            def do_PATCH(self) -> None:
                proxy._handle(self)

            def do_DELETE(self) -> None:
                proxy._handle(self)

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

        self._server = ThreadingHTTPServer((self.options.host, self.options.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="agentqa-fault-proxy",
            daemon=True,
        )
        self._thread.start()
        return self.endpoint

    def serve_forever(self) -> None:
        """Run in the foreground until interrupted."""
        if self._server is not None:
            raise RuntimeError("fault proxy is already running")
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                proxy._handle(self)

            def do_POST(self) -> None:
                proxy._handle(self)

            def do_PUT(self) -> None:
                proxy._handle(self)

            def do_PATCH(self) -> None:
                proxy._handle(self)

            def do_DELETE(self) -> None:
                proxy._handle(self)

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

        self._server = ThreadingHTTPServer((self.options.host, self.options.port), Handler)
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()
            self._server = None
            self._cassettes.flush()
            self._client.close()

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._cassettes.flush()
        self._client.close()

    def wait(self) -> None:
        """Block until a background server exits."""
        if self._thread is None:
            raise RuntimeError("fault proxy is not running in the background")
        self._thread.join()

    @staticmethod
    def _send_json(
        handler: BaseHTTPRequestHandler,
        status_code: int,
        code: str,
    ) -> None:
        body = canonical_json_bytes({"error": code})
        handler.send_response(status_code)
        handler.send_header("content-type", "application/json")
        handler.send_header("content-length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _read_request_body(self, handler: BaseHTTPRequestHandler) -> bytes | None:
        if handler.headers.get("transfer-encoding"):
            self._send_json(handler, 400, "CHUNKED_REQUEST_NOT_SUPPORTED")
            return None
        try:
            length = int(handler.headers.get("content-length", "0"))
        except ValueError:
            self._send_json(handler, 400, "INVALID_CONTENT_LENGTH")
            return None
        if length < 0 or length > self.options.max_request_bytes:
            self._send_json(handler, 413, "REQUEST_TOO_LARGE")
            return None
        return handler.rfile.read(length)

    def _request_target(self, raw_target: str) -> tuple[str, str] | None:
        parsed = urlsplit(raw_target)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            return None
        decoded_path = unquote(parsed.path)
        if ".." in decoded_path.split("/") or decoded_path not in self.options.allowed_paths:
            return None
        upstream = urlsplit(self.options.upstream_base_url)
        base_path = upstream.path.rstrip("/")
        path = f"{base_path}{parsed.path}" if base_path else parsed.path
        url = urlunsplit((upstream.scheme, upstream.netloc, path, parsed.query, ""))
        path_and_query = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        return url, path_and_query

    @staticmethod
    def _outbound_headers(handler: BaseHTTPRequestHandler) -> dict[str, str]:
        return {
            name: value
            for name, value in handler.headers.items()
            if name.lower()
            not in _HOP_BY_HOP_HEADERS | _CONTROL_HEADERS | {"host", "content-length"}
        }

    def _upstream(
        self,
        handler: BaseHTTPRequestHandler,
        url: str,
        body: bytes,
    ) -> _ProxyResponse:
        request = self._client.build_request(
            handler.command,
            url,
            headers=self._outbound_headers(handler),
            content=body,
        )
        response = self._client.send(request, stream=True)
        try:
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > self.options.max_response_bytes:
                    raise FaultProxyError("upstream response exceeded max_response_bytes")
                chunks.append(chunk)
            headers = {
                name: value
                for name, value in response.headers.items()
                if name.lower() not in _HOP_BY_HOP_HEADERS | {"content-encoding", "content-length"}
            }
            return _ProxyResponse(response.status_code, headers, b"".join(chunks), "upstream")
        finally:
            response.close()

    def _baseline(
        self,
        handler: BaseHTTPRequestHandler,
        url: str,
        path_and_query: str,
        body: bytes,
    ) -> _ProxyResponse:
        arguments, call_index = self._cassettes.request(
            handler.command,
            path_and_query,
            {name: value for name, value in handler.headers.items()},
            body,
        )
        if self.options.mode == "replay":
            replay = self._cassettes.replay(arguments, call_index)
            if replay is not None:
                return replay
            if not self._cassettes.allow_live_fallback:
                raise FaultProxyError("cassette miss")
        trace_id = handler.headers.get("x-agentqa-trace-id", "proxy-trace")
        span_id = handler.headers.get("x-agentqa-span-id", secrets.token_hex(8))
        try:
            response = self._upstream(handler, url, body)
        except (FaultProxyError, httpx.HTTPError) as exc:
            self._cassettes.record_error(
                arguments,
                call_index,
                exc,
                trace_id=trace_id,
                span_id=span_id,
            )
            raise
        self._cassettes.record(
            arguments,
            call_index,
            response,
            trace_id=trace_id,
            span_id=span_id,
        )
        return response

    def _session(
        self,
        handler: BaseHTTPRequestHandler,
        mutation_id: str,
    ) -> MutationTrialSession:
        trial_id = handler.headers.get("x-agentqa-trial-id", "proxy-trial")
        test_id = handler.headers.get("x-agentqa-test-id", "proxy-test")
        trace_id = handler.headers.get("x-agentqa-trace-id", "proxy-trace")
        session_key = f"{trial_id}\0{test_id}\0{trace_id}\0{mutation_id}"
        try:
            seed = int(handler.headers.get("x-agentqa-trial-seed", "0"))
        except ValueError:
            seed = 0
        try:
            labels_value = json.loads(handler.headers.get("x-agentqa-test-labels", "{}"))
            labels = (
                {str(key): str(value) for key, value in labels_value.items()}
                if isinstance(labels_value, dict)
                else {}
            )
        except json.JSONDecodeError:
            labels = {}
        with self._session_lock:
            session = self._sessions.get(session_key)
            if session is None:
                session = MutationTrialSession(
                    RunContext(
                        run_id="fault-proxy",
                        spec_hash="fault-proxy",
                        started_at=utc_now(),
                        random_seed=seed,
                        test_case_id=test_id,
                        trial_id=trial_id,
                        trace_id=trace_id,
                        mutation_id=mutation_id,
                        trial_seed=seed,
                        test_labels=labels,
                    )
                )
                self._sessions[session_key] = session
            return session

    @staticmethod
    def _arguments(
        handler: BaseHTTPRequestHandler,
        path_and_query: str,
        body: bytes,
    ) -> dict[str, Any]:
        try:
            parsed_body: Any = json.loads(body) if body else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed_body = "<non-json-body>"
        return {
            "method": handler.command,
            "path": urlsplit(path_and_query).path,
            "query": urlsplit(path_and_query).query,
            "body": parsed_body,
        }

    @staticmethod
    def _mutate_response(
        spec: MutationSpec,
        baseline: _ProxyResponse,
    ) -> tuple[_ProxyResponse, bool, dict[str, Any]]:
        operation = spec.operation
        if operation.type == "invalid_json":
            return (
                _ProxyResponse(
                    baseline.status_code,
                    {**baseline.headers, "content-type": "application/json"},
                    b"{agentqa-invalid-json",
                    "synthetic",
                ),
                True,
                {"replacement": "invalid_json"},
            )
        try:
            payload = json.loads(baseline.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return baseline, False, {"reason": "upstream_response_not_json"}
        if operation.type == "missing_field":
            updated, changed = delete_path(payload, operation.field or "")
            behavior = {"field": operation.field or "", "deleted": changed}
        else:
            updated, changed = replace_path(payload, operation.field or "", operation.value)
            behavior = {"field": operation.field or "", "replaced": changed}
        if not changed:
            return baseline, False, behavior
        return (
            _ProxyResponse(
                baseline.status_code,
                {**baseline.headers, "content-type": "application/json"},
                canonical_json_bytes(updated),
                "synthetic",
            ),
            True,
            behavior,
        )

    def _apply(
        self,
        handler: BaseHTTPRequestHandler,
        spec: MutationSpec,
        baseline_call: Callable[[], _ProxyResponse],
    ) -> tuple[_ProxyResponse | None, bool, dict[str, Any]]:
        operation = spec.operation
        if operation.type == "timeout":
            time.sleep((operation.delay_ms or 1) / 1000)
            return baseline_call(), True, {"delay_ms": operation.delay_ms or 1}
        if operation.type == "http_status":
            body = canonical_json_bytes({"error": "AGENTQA_INJECTED_HTTP_STATUS"})
            return (
                _ProxyResponse(
                    operation.status_code or 503,
                    {"content-type": "application/json"},
                    body,
                    "synthetic",
                ),
                True,
                {"status_code": operation.status_code or 503},
            )
        if operation.type == "http_disconnect":
            return None, True, {"action": "disconnect"}
        baseline = baseline_call()
        return self._mutate_response(spec, baseline)

    def _record_application(
        self,
        handler: BaseHTTPRequestHandler,
        spec: MutationSpec,
        session: MutationTrialSession,
        call_index: int,
        match: MutationMatch,
        arguments: dict[str, Any],
        response: _ProxyResponse | None,
        triggered: bool,
        behavior: dict[str, Any],
    ) -> None:
        context = session.run_context
        application = MutationApplication(
            mutation_id=spec.metadata.name,
            operation_type=spec.operation.type,
            interception=InterceptionPoint(kind="http", name=self.options.route_name),
            call_index=call_index,
            triggered=triggered,
            match=match,
            invocation_summary=summarize(arguments),
            result_summary=(
                None
                if response is None
                else summarize(
                    {
                        "status_code": response.status_code,
                        "body": response.body.decode("utf-8", "replace"),
                    }
                )
            ),
            behavior=cast(dict[str, Any], safe_jsonable(behavior)),
            test_case_id=context.test_case_id or "proxy-test",
            trial_id=context.trial_id or "proxy-trial",
            trace_id=context.trace_id or "proxy-trace",
            span_id=handler.headers.get("x-agentqa-span-id", secrets.token_hex(8)),
            applied_at=utc_now(),
        )
        self._cassettes.append_application(application)

    def _send_response(
        self,
        handler: BaseHTTPRequestHandler,
        response: _ProxyResponse,
        mutation_id: str | None,
        triggered: bool,
    ) -> None:
        handler.send_response(response.status_code)
        for name, value in response.headers.items():
            if name.lower() not in _HOP_BY_HOP_HEADERS | {"content-length"}:
                handler.send_header(name, value)
        handler.send_header("content-length", str(len(response.body)))
        handler.send_header("x-agentqa-proxy-source", response.source)
        if mutation_id is not None:
            handler.send_header("x-agentqa-mutation-id", mutation_id)
            handler.send_header("x-agentqa-mutation-triggered", str(triggered).lower())
        handler.end_headers()
        handler.wfile.write(response.body)

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        target = self._request_target(handler.path)
        if target is None:
            self._send_json(handler, 403, "PATH_NOT_ALLOWED")
            return
        body = self._read_request_body(handler)
        if body is None:
            return
        url, path_and_query = target
        mutation_id = handler.headers.get("x-agentqa-mutation-id")
        if mutation_id is None:
            try:
                response = self._baseline(handler, url, path_and_query, body)
            except (FaultProxyError, httpx.HTTPError):
                self._send_json(handler, 502, "UPSTREAM_OR_CASSETTE_ERROR")
                return
            self._send_response(handler, response, None, False)
            return
        spec = self._mutations.get(mutation_id)
        if spec is None:
            self._send_json(handler, 400, "UNKNOWN_HTTP_MUTATION")
            return

        session = self._session(handler, mutation_id)
        interception = InterceptionPoint(kind="http", name=self.options.route_name)
        with self._session_lock:
            call_index = session.next_call_index(interception.kind, interception.name)
            arguments = self._arguments(handler, path_and_query, body)
            match = match_mutation(spec, interception, arguments, call_index, session)
        try:
            if match.matched:
                mutation_response, triggered, behavior = self._apply(
                    handler,
                    spec,
                    lambda: self._baseline(handler, url, path_and_query, body),
                )
            else:
                mutation_response = self._baseline(handler, url, path_and_query, body)
                triggered = False
                behavior = {"action": "pass_through"}
        except (FaultProxyError, httpx.HTTPError) as exc:
            error_triggered = match.matched and spec.operation.type == "timeout"
            if error_triggered:
                with self._session_lock:
                    session.applied_mutations.add(mutation_id)
            self._record_application(
                handler,
                spec,
                session,
                call_index,
                match,
                arguments,
                None,
                error_triggered,
                {"action": "proxy_error", "error_type": type(exc).__name__},
            )
            self._send_json(handler, 502, "UPSTREAM_OR_CASSETTE_ERROR")
            return
        if triggered:
            with self._session_lock:
                session.applied_mutations.add(mutation_id)
        self._record_application(
            handler,
            spec,
            session,
            call_index,
            match,
            arguments,
            mutation_response,
            triggered,
            behavior,
        )
        if mutation_response is None:
            handler.close_connection = True
            with suppress(OSError):
                handler.connection.shutdown(2)
            handler.connection.close()
            return
        self._send_response(handler, mutation_response, mutation_id, triggered)
