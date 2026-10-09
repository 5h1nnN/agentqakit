"""Dependency-neutral conversion of framework values to bounded JSON data."""

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, JsonValue


def to_jsonable(value: Any, *, depth: int = 0) -> Any:
    """Normalize common framework objects to bounded JSON-compatible data."""
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
    content = getattr(value, "content", None)
    if content is not None:
        result: dict[str, JsonValue] = {
            "type": type(value).__name__,
            "content": to_jsonable(content, depth=depth + 1),
        }
        name = getattr(value, "name", None)
        if isinstance(name, str) and name:
            result["name"] = name
        return result
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"
