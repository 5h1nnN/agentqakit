"""Mutation-safe JSON conversion and restricted field paths."""

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel, JsonValue

from agentqa.storage import redact_secrets
from agentqa.tracing.normalization import content_value
from agentqa.tracing.options import TraceOptions

_PATH_PART = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)|\[(-?\d+)\]")
_MISSING = object()


def to_jsonable(value: Any, *, depth: int = 0) -> JsonValue:
    """Convert runtime values into bounded, redacted JSON data."""
    if depth > 30:
        return "<maximum-depth>"
    if isinstance(value, float) and not math.isfinite(value):
        return "<non-finite-float>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"), depth=depth + 1)
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item, depth=depth + 1)
            for key, item in list(value.items())[:10_000]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item, depth=depth + 1) for item in list(value)[:10_000]]
    page_content = getattr(value, "page_content", None)
    if isinstance(page_content, str):
        metadata = getattr(value, "metadata", {})
        return {
            "type": type(value).__name__,
            "page_content": page_content,
            "metadata": to_jsonable(metadata, depth=depth + 1),
        }
    content = getattr(value, "content", None)
    if content is not None:
        return {
            "type": type(value).__name__,
            "content": to_jsonable(content, depth=depth + 1),
        }
    try:
        return cast(JsonValue, json.loads(json.dumps(value)))
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"


def safe_jsonable(value: Any) -> JsonValue:
    """Convert and redact a runtime value."""
    return cast(JsonValue, redact_secrets(to_jsonable(value)))


def summarize(value: Any) -> JsonValue:
    """Return a non-reversible summary regardless of trace content settings."""
    return content_value(safe_jsonable(value), TraceOptions())


def path_tokens(path: str) -> list[str | int]:
    """Parse a restricted dotted/index path without expressions."""
    value = path.strip()
    if value.startswith("$."):
        value = value[2:]
    elif value.startswith("$"):
        value = value[1:]
    if value.startswith("."):
        value = value[1:]
    if not value:
        return []
    tokens: list[str | int] = []
    position = 0
    while position < len(value):
        if value[position] == ".":
            position += 1
            continue
        match = _PATH_PART.match(value, position)
        if match is None:
            raise ValueError("field path supports only names and integer indexes")
        name, index = match.groups()
        tokens.append(name if name is not None else int(index))
        position = match.end()
    return tokens


def _path_child(value: Any, token: str | int) -> tuple[Any, bool]:
    if isinstance(token, str) and isinstance(value, Mapping) and token in value:
        return value[token], True
    if (
        isinstance(token, int)
        and isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and -len(value) <= token < len(value)
    ):
        return value[token], True
    return None, False


def get_path(value: Any, path: str) -> Any:
    """Return a restricted path value or a private missing sentinel."""
    current = value
    for token in path_tokens(path):
        current, found = _path_child(current, token)
        if not found:
            return _MISSING
    return current


def path_exists(value: Any, path: str) -> bool:
    """Return whether a restricted field path exists."""
    return get_path(value, path) is not _MISSING


def delete_path(value: Any, path: str) -> tuple[Any, bool]:
    """Deep-copy *value* and delete an existing field or list item."""
    result = copy.deepcopy(value)
    tokens = path_tokens(path)
    if not tokens:
        return result, False
    parent = result
    for token in tokens[:-1]:
        parent, found = _path_child(parent, token)
        if not found:
            return result, False
    leaf = tokens[-1]
    if isinstance(leaf, str) and isinstance(parent, dict) and leaf in parent:
        del parent[leaf]
        return result, True
    if isinstance(leaf, int) and isinstance(parent, list) and -len(parent) <= leaf < len(parent):
        del parent[leaf]
        return result, True
    return result, False


def replace_path(value: Any, path: str, replacement: JsonValue) -> tuple[Any, bool]:
    """Deep-copy *value* and replace one existing field or list item."""
    result = copy.deepcopy(value)
    tokens = path_tokens(path)
    if not tokens:
        return replacement, True
    parent = result
    for token in tokens[:-1]:
        parent, found = _path_child(parent, token)
        if not found:
            return result, False
    leaf = tokens[-1]
    if isinstance(leaf, str) and isinstance(parent, dict) and leaf in parent:
        parent[leaf] = replacement
        return result, True
    if isinstance(leaf, int) and isinstance(parent, list) and -len(parent) <= leaf < len(parent):
        parent[leaf] = replacement
        return result, True
    return result, False
