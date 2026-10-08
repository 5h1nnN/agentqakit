"""Canonical serialization and stable specification hashing."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel


def canonical_json_bytes(value: Any, *, by_alias: bool = False) -> bytes:
    """Encode JSON-compatible data deterministically as UTF-8."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=by_alias)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_digest(value: bytes) -> str:
    """Return a namespaced SHA-256 digest."""
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def normalize_spec(spec: BaseModel) -> dict[str, Any]:
    """Return a validated spec with canonical external camelCase fields."""
    return spec.model_dump(mode="json", by_alias=True)


def hash_spec(spec: BaseModel) -> str:
    """Hash a validated spec using its canonical camelCase representation."""
    return sha256_digest(canonical_json_bytes(normalize_spec(spec)))


def hash_spec_collection(spec_hashes: Sequence[str]) -> str:
    """Hash an ordered collection of individual specification hashes."""
    return sha256_digest(canonical_json_bytes(list(spec_hashes)))
