"""Restricted, non-executable template rendering."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentqa.errors import AgentQAError

_TOKEN = re.compile(
    r"{{\s*(?P<namespace>input|env)\."
    r"(?P<path>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*}}"
)
_MAX_TEMPLATE_NODES = 100_000
_MAX_TEMPLATE_DEPTH = 100


class TemplateRenderError(AgentQAError):
    """Raised when a restricted template is invalid or unresolved."""

    def __init__(self, message: str, *, path: tuple[str | int, ...]) -> None:
        super().__init__(message, code="TEMPLATE_ERROR", details={"path": list(path)})
        self.path = path


@dataclass(frozen=True, slots=True)
class RenderedTemplate:
    """Actual rendered data plus a structurally equivalent redacted view."""

    value: Any
    redacted: Any


def _lookup_input(values: Mapping[str, Any], dotted_path: str, path: tuple[str | int, ...]) -> Any:
    current: Any = values
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise TemplateRenderError(
                f"missing template variable input.{dotted_path}",
                path=path,
            )
        current = current[part]
    return current


def _resolve_token(
    namespace: str,
    token_path: str,
    *,
    input_values: Mapping[str, Any],
    environment: Mapping[str, str],
    resolve_input: bool,
    path: tuple[str | int, ...],
) -> tuple[Any, Any, bool]:
    if namespace == "input":
        if not resolve_input:
            return None, None, False
        value = _lookup_input(input_values, token_path, path)
        return value, value, True

    if "." in token_path:
        raise TemplateRenderError(
            "environment variable names cannot contain dots",
            path=path,
        )
    if token_path not in environment:
        raise TemplateRenderError(
            f"missing environment variable {token_path}",
            path=path,
        )
    return environment[token_path], "***", True


def _render_string(
    value: str,
    *,
    input_values: Mapping[str, Any],
    environment: Mapping[str, str],
    resolve_input: bool,
    path: tuple[str | int, ...],
) -> RenderedTemplate:
    matches = list(_TOKEN.finditer(value))
    if not matches:
        if "{{" in value or "}}" in value:
            raise TemplateRenderError("unsupported template expression", path=path)
        return RenderedTemplate(value=value, redacted=value)

    unmatched = _TOKEN.sub("", value)
    if "{{" in unmatched or "}}" in unmatched:
        raise TemplateRenderError("unsupported template expression", path=path)

    full_match = _TOKEN.fullmatch(value)
    if full_match is not None:
        actual, redacted, resolved = _resolve_token(
            full_match.group("namespace"),
            full_match.group("path"),
            input_values=input_values,
            environment=environment,
            resolve_input=resolve_input,
            path=path,
        )
        if resolved:
            return RenderedTemplate(value=actual, redacted=redacted)
        return RenderedTemplate(value=value, redacted=value)

    actual_parts: list[str] = []
    redacted_parts: list[str] = []
    cursor = 0
    for match in matches:
        prefix = value[cursor : match.start()]
        actual_parts.append(prefix)
        redacted_parts.append(prefix)
        actual, redacted, resolved = _resolve_token(
            match.group("namespace"),
            match.group("path"),
            input_values=input_values,
            environment=environment,
            resolve_input=resolve_input,
            path=path,
        )
        if resolved:
            actual_parts.append(str(actual))
            redacted_parts.append(str(redacted))
        else:
            actual_parts.append(match.group(0))
            redacted_parts.append(match.group(0))
        cursor = match.end()
    suffix = value[cursor:]
    actual_parts.append(suffix)
    redacted_parts.append(suffix)
    return RenderedTemplate(value="".join(actual_parts), redacted="".join(redacted_parts))


def _render_value(
    value: Any,
    *,
    input_values: Mapping[str, Any],
    environment: Mapping[str, str],
    resolve_input: bool,
    path: tuple[str | int, ...],
    ancestors: frozenset[int],
    visited_nodes: list[int],
) -> RenderedTemplate:
    visited_nodes[0] += 1
    if visited_nodes[0] > _MAX_TEMPLATE_NODES:
        raise TemplateRenderError("template data is too large", path=path)
    if len(path) > _MAX_TEMPLATE_DEPTH:
        raise TemplateRenderError("template data is too deeply nested", path=path)
    if isinstance(value, str):
        return _render_string(
            value,
            input_values=input_values,
            environment=environment,
            resolve_input=resolve_input,
            path=path,
        )
    if isinstance(value, list):
        identity = id(value)
        if identity in ancestors:
            raise TemplateRenderError("template data contains a reference cycle", path=path)
        child_ancestors = ancestors | {identity}
        rendered = [
            _render_value(
                item,
                input_values=input_values,
                environment=environment,
                resolve_input=resolve_input,
                path=(*path, index),
                ancestors=child_ancestors,
                visited_nodes=visited_nodes,
            )
            for index, item in enumerate(value)
        ]
        return RenderedTemplate(
            value=[item.value for item in rendered],
            redacted=[item.redacted for item in rendered],
        )
    if isinstance(value, dict):
        identity = id(value)
        if identity in ancestors:
            raise TemplateRenderError("template data contains a reference cycle", path=path)
        child_ancestors = ancestors | {identity}
        rendered_mapping = {
            key: _render_value(
                item,
                input_values=input_values,
                environment=environment,
                resolve_input=resolve_input,
                path=(*path, str(key)),
                ancestors=child_ancestors,
                visited_nodes=visited_nodes,
            )
            for key, item in value.items()
        }
        return RenderedTemplate(
            value={key: item.value for key, item in rendered_mapping.items()},
            redacted={key: item.redacted for key, item in rendered_mapping.items()},
        )
    return RenderedTemplate(value=value, redacted=value)


def render_templates(
    value: Any,
    *,
    input_values: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    resolve_input: bool = True,
) -> RenderedTemplate:
    """Render only ``input`` and ``env`` lookups without evaluating code."""
    return _render_value(
        value,
        input_values=input_values or {},
        environment=os.environ if environment is None else environment,
        resolve_input=resolve_input,
        path=(),
        ancestors=frozenset(),
        visited_nodes=[0],
    )
