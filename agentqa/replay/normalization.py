"""Canonical, privacy-aware interaction normalization."""

import re
from copy import deepcopy
from typing import Any, cast

from pydantic import JsonValue

from agentqa.json import to_jsonable
from agentqa.storage import redact_secrets, redact_text_secrets

_PATH_PART = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)|\[(-?\d+)\]")


def _redact_text_values(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return redact_text_secrets(value)
    if isinstance(value, list):
        return [_redact_text_values(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_text_values(item) for key, item in value.items()}
    return value


def safe_value(value: Any) -> JsonValue:
    """Convert arbitrary framework data to bounded JSON and redact secrets."""
    normalized = cast(JsonValue, to_jsonable(value))
    redacted = cast(JsonValue, redact_secrets(normalized))
    return _redact_text_values(redacted)


def _path_tokens(path: str) -> list[str | int]:
    value = path.strip()
    if value == "$":
        return []
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
            raise ValueError("volatile field paths support only names and integer indexes")
        name, index = match.groups()
        tokens.append(name if name is not None else int(cast(str, index)))
        position = match.end()
    return tokens


def _remove_path(value: JsonValue, path: str) -> None:
    tokens = _path_tokens(path)
    if not tokens:
        return
    current: Any = value
    for token in tokens[:-1]:
        if isinstance(token, str) and isinstance(current, dict):
            current = current.get(token)
        elif isinstance(token, int) and isinstance(current, list):
            if not -len(current) <= token < len(current):
                return
            current = current[token]
        else:
            return
    final = tokens[-1]
    if isinstance(final, str) and isinstance(current, dict):
        current.pop(final, None)
    elif (
        isinstance(final, int)
        and isinstance(current, list)
        and -len(current) <= final < len(current)
    ):
        current.pop(final)


def normalize_arguments(
    arguments: Any,
    volatile_fields: tuple[str, ...] = (),
) -> JsonValue:
    """Redact and remove declared volatile paths before matching or hashing."""
    normalized = deepcopy(safe_value(arguments))
    for path in volatile_fields:
        _remove_path(normalized, path)
    return normalized
