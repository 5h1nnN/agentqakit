"""Internal helpers shared by target adapters."""

import hashlib
import importlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from agentqa.errors import AgentQAError
from agentqa.json import to_jsonable as _to_jsonable
from agentqa.storage import redact_secrets
from agentqa.targets.errors import TargetConfigurationError

_PATH_PART = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)|\[(-?\d+)\]")


def to_jsonable(value: Any, *, depth: int = 0) -> Any:
    """Normalize common framework objects to bounded JSON-compatible data."""
    return _to_jsonable(value, depth=depth)


def load_entrypoint(entrypoint: str) -> Any:
    """Import a validated ``module:attribute`` reference."""
    module_name, attribute = entrypoint.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing.startswith(("langgraph", "langchain")):
            raise TargetConfigurationError(
                "LangGraph dependencies are not installed; install agentqakit[langgraph]"
            ) from None
        if exc.name == module_name:
            raise TargetConfigurationError(
                f"cannot import {module_name}; install the target's optional dependencies"
            ) from None
        raise TargetConfigurationError(
            f"dependency {exc.name!r} required by {module_name} is not installed"
        ) from None
    except Exception as exc:
        raise TargetConfigurationError(
            f"importing {module_name} raised {type(exc).__name__}"
        ) from None
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise TargetConfigurationError(
            f"module {module_name} has no attribute {attribute}"
        ) from None


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
            raise ValueError("output path supports only names and integer indexes")
        name, index = match.groups()
        tokens.append(name if name is not None else int(cast(str, index)))
        position = match.end()
    return tokens


def extract_path(value: Any, path: str) -> Any:
    """Resolve a restricted JSONPath/dotted path without executing expressions."""
    current = value
    for token in _path_tokens(path):
        if isinstance(token, str) and isinstance(current, Mapping) and token in current:
            current = current[token]
        elif (
            isinstance(token, int)
            and isinstance(current, Sequence)
            and not isinstance(current, (str, bytes, bytearray))
        ):
            try:
                current = current[token]
            except IndexError:
                raise KeyError(path) from None
        else:
            raise KeyError(path)
    return current


def trace_id_hex(value: str | None) -> str:
    """Return a W3C-compatible 32-hex trace ID."""
    compact = "" if value is None else value.replace("-", "").lower()
    if len(compact) == 32 and compact != "0" * 32 and all(c in "0123456789abcdef" for c in compact):
        return compact
    return hashlib.sha256((value or "agentqa").encode()).hexdigest()[:32]


def safe_jsonable(value: Any) -> Any:
    """Normalize and redact credential-like fields."""
    return redact_secrets(to_jsonable(value))


def safe_error_details(exc: BaseException) -> dict[str, Any]:
    """Return non-sensitive structured details for expected adapter errors."""
    if isinstance(exc, AgentQAError):
        redacted = redact_secrets(exc.details)
        return cast(dict[str, Any], redacted)
    return {}
