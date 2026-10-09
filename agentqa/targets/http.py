"""Safe HTTP target adapter."""

import asyncio
import ipaddress
import json
import secrets
import socket
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from opentelemetry import propagate

from agentqa.conventions import utc_now
from agentqa.models import ErrorInfo, HttpTargetSpec, RunContext, TargetResult, TestInput
from agentqa.specs import TemplateRenderError, render_templates
from agentqa.storage import redact_secrets
from agentqa.targets.errors import TargetErrorCode, TargetSafetyError
from agentqa.targets.utils import extract_path, safe_error_details, trace_id_hex

_LOCAL_HOSTS = {"localhost", "localhost.localdomain"}
_METADATA_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.azure.internal",
}
_FORBIDDEN_REQUEST_HEADERS = {"connection", "content-length", "host", "transfer-encoding"}


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _is_forbidden_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not address.is_loopback and (
        address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or str(address) == "169.254.169.254"
    )


class HttpTargetAdapter:
    """Invoke a JSON HTTP endpoint within explicit SSRF and resource limits."""

    def __init__(
        self,
        spec: HttpTargetSpec,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._spec = spec
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def prepare(self, context: RunContext) -> None:
        del context
        await self._validate_destination(str(self._spec.url))
        timeout = httpx.Timeout(
            timeout=self._spec.timeout_ms / 1000,
            connect=self._spec.connect_timeout_ms / 1000,
            read=self._spec.read_timeout_ms / 1000,
        )
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
        )

    async def cleanup(self, context: RunContext) -> None:
        del context
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _resolve(
        self, hostname: str, port: int
    ) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        loop = asyncio.get_running_loop()
        try:
            records = await asyncio.wait_for(
                loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM),
                timeout=self._spec.connect_timeout_ms / 1000,
            )
        except (TimeoutError, socket.gaierror) as exc:
            raise TargetSafetyError(
                f"cannot safely resolve target host: {type(exc).__name__}"
            ) from None
        return {ipaddress.ip_address(record[4][0]) for record in records}

    async def _validate_destination(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise TargetSafetyError("target URL must use http or https and include a host")
        hostname = parsed.hostname.rstrip(".").lower()
        if parsed.username is not None or parsed.password is not None:
            raise TargetSafetyError("credentials are not allowed in target URLs")
        if hostname in _METADATA_HOSTS:
            raise TargetSafetyError("cloud metadata endpoints are always blocked")
        allowed_hosts = {host.rstrip(".").lower() for host in self._spec.allowed_hosts}
        if allowed_hosts and hostname not in allowed_hosts:
            raise TargetSafetyError("target host is not present in allowedHosts")

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = await self._resolve(hostname, port)
        if not addresses:
            raise TargetSafetyError("target host did not resolve to an address")
        if any(_is_forbidden_address(address) for address in addresses):
            raise TargetSafetyError("target host resolves to a forbidden network address")

        all_loopback = all(address.is_loopback for address in addresses)
        try:
            literal_address = ipaddress.ip_address(hostname)
        except ValueError:
            literal_address = None
        local_identity = hostname in _LOCAL_HOSTS or (
            literal_address is not None and literal_address.is_loopback
        )
        if not self._spec.allow_remote and (not local_identity or not all_loopback):
            raise TargetSafetyError("remote HTTP targets require allowRemote: true")
        if (
            self._spec.allow_remote
            and not self._spec.allow_private_network
            and (
                any(address.is_private and not address.is_loopback for address in addresses)
                or (not local_identity and any(address.is_loopback for address in addresses))
            )
        ):
            raise TargetSafetyError(
                "private-network HTTP targets require allowPrivateNetwork: true"
            )

    @staticmethod
    def _error_result(
        *,
        started_at: Any,
        code: TargetErrorCode,
        message: str,
        status: Literal["ERROR", "TIMEOUT"] = "ERROR",
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TargetResult:
        return TargetResult(
            status=status,
            error=ErrorInfo(code=code.value, message=message, details=details or {}),
            started_at=started_at,
            ended_at=utc_now(),
            metadata=metadata or {},
        )

    def _render_request(self, test_input: TestInput) -> tuple[dict[str, str], dict[str, str], Any]:
        rendered = render_templates(
            {
                "headers": self._spec.request.headers,
                "query": self._spec.request.query,
                "json": self._spec.request.json_body,
            },
            input_values=test_input.input,
        ).value
        if not isinstance(rendered, Mapping):
            raise TemplateRenderError("rendered HTTP request must be a mapping", path=())
        headers = rendered.get("headers")
        query = rendered.get("query")
        if not isinstance(headers, Mapping) or not all(
            isinstance(key, str) and isinstance(value, (str, int, float, bool))
            for key, value in headers.items()
        ):
            raise TemplateRenderError(
                "rendered HTTP headers must be scalar values", path=("headers",)
            )
        if any(str(key).lower() in _FORBIDDEN_REQUEST_HEADERS for key in headers):
            raise TemplateRenderError(
                "unsafe HTTP request header is not allowed", path=("headers",)
            )
        if not isinstance(query, Mapping) or not all(
            isinstance(key, str) and isinstance(value, (str, int, float, bool))
            for key, value in query.items()
        ):
            raise TemplateRenderError("rendered HTTP query values must be scalar", path=("query",))
        return (
            {str(key): str(value) for key, value in headers.items()},
            {str(key): str(value) for key, value in query.items()},
            rendered.get("json"),
        )

    @staticmethod
    def _trace_headers(context: RunContext) -> dict[str, str]:
        trace_id = trace_id_hex(context.trace_id)
        headers = {
            "x-agentqa-run-id": context.run_id,
            "x-agentqa-test-id": context.test_case_id or "unknown",
            "x-agentqa-trial-id": context.trial_id or "unknown",
            "x-agentqa-trace-id": context.trace_id or trace_id,
            "x-agentqa-test-labels": json.dumps(
                context.test_labels,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        if context.trial_seed is not None:
            headers["x-agentqa-trial-seed"] = str(context.trial_seed)
        if context.mutation_id is not None:
            headers["x-agentqa-mutation-id"] = context.mutation_id
        propagate.inject(headers)
        headers.setdefault("traceparent", f"00-{trace_id}-{secrets.token_hex(8)}-01")
        return headers

    async def invoke(self, test_input: TestInput, context: RunContext) -> TargetResult:
        started_at = utc_now()
        if self._client is None:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.HTTP_CONNECTION_ERROR,
                message="HTTP adapter has not been prepared",
            )
        try:
            url = str(self._spec.url)
            await self._validate_destination(url)
            headers, query, json_body = self._render_request(test_input)
            headers.update(self._trace_headers(context))
            request_args: dict[str, Any] = {
                "headers": headers,
                "params": query,
            }
            if self._spec.method == "POST":
                request_args["json"] = json_body

            async with self._client.stream(self._spec.method, url, **request_args) as response:
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._spec.max_response_bytes:
                        return self._error_result(
                            started_at=started_at,
                            code=TargetErrorCode.HTTP_RESPONSE_TOO_LARGE,
                            message="HTTP response exceeded maxResponseBytes",
                            details={"limit_bytes": self._spec.max_response_bytes},
                            metadata={"status_code": response.status_code},
                        )

                allowed = {name.lower() for name in self._spec.response.allowed_headers}
                response_headers = {
                    name: value
                    for name, value in response.headers.items()
                    if name.lower() in allowed
                }
                metadata: dict[str, Any] = {
                    "status_code": response.status_code,
                    "response_bytes": len(body),
                    "headers": redact_secrets(response_headers),
                    "latency_ms": (utc_now() - started_at).total_seconds() * 1000,
                }
                document: Any = None
                invalid_json = False
                if body:
                    try:
                        document = json.loads(
                            body,
                            parse_constant=_reject_nonstandard_json_constant,
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                        invalid_json = True
                    else:
                        metadata["body"] = redact_secrets(document)
                if response.status_code >= 300:
                    return self._error_result(
                        started_at=started_at,
                        code=TargetErrorCode.HTTP_STATUS_ERROR,
                        message=f"HTTP target returned status {response.status_code}",
                        details={"status_code": response.status_code},
                        metadata=metadata,
                    )
                if not body:
                    return self._error_result(
                        started_at=started_at,
                        code=TargetErrorCode.HTTP_INVALID_RESPONSE,
                        message="HTTP target returned an empty response",
                        metadata=metadata,
                    )
                if invalid_json:
                    return self._error_result(
                        started_at=started_at,
                        code=TargetErrorCode.HTTP_INVALID_RESPONSE,
                        message="HTTP target returned a non-JSON response",
                        metadata=metadata,
                    )
                try:
                    output = extract_path(document, self._spec.response.output_path)
                except (KeyError, ValueError):
                    return self._error_result(
                        started_at=started_at,
                        code=TargetErrorCode.HTTP_OUTPUT_ERROR,
                        message="configured response outputPath was not found",
                        metadata=metadata,
                    )
                return TargetResult(
                    status="OK",
                    output=output,
                    started_at=started_at,
                    ended_at=utc_now(),
                    metadata=metadata,
                )
        except TemplateRenderError as exc:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.CONFIGURATION_ERROR,
                message=exc.message,
                details=safe_error_details(exc),
            )
        except TargetSafetyError as exc:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.HTTP_SAFETY_ERROR,
                message=exc.message,
            )
        except httpx.TimeoutException:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.HTTP_TIMEOUT,
                message="HTTP target timed out",
                status="TIMEOUT",
            )
        except httpx.RequestError as exc:
            return self._error_result(
                started_at=started_at,
                code=TargetErrorCode.HTTP_CONNECTION_ERROR,
                message=f"HTTP request failed with {type(exc).__name__}",
            )
