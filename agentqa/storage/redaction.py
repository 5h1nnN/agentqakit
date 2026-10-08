"""Conservative secret redaction before artifact serialization."""

import re
from typing import Any

from pydantic import BaseModel

REDACTED = "***"
_SENSITIVE_KEYS = {
    "accesstoken",
    "apikey",
    "authorization",
    "clientsecret",
    "cookie",
    "password",
    "refreshtoken",
    "secret",
    "setcookie",
    "token",
}


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def is_sensitive_key(value: str) -> bool:
    """Return whether a mapping key conventionally contains credentials."""
    normalized = _normalize_key(value)
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith("apikey")
        or normalized.endswith("password")
    )


def redact_secrets(value: Any) -> Any:
    """Recursively replace values under credential-like keys."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): REDACTED if is_sensitive_key(str(key)) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value
