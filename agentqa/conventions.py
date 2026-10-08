"""Shared time and identifier conventions."""

from datetime import UTC, datetime
from uuid import UUID, uuid4


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def to_utc_iso8601(value: datetime) -> str:
    """Serialize an aware datetime as UTC ISO 8601 with a ``Z`` suffix."""
    if value.tzinfo is None or value.utcoffset() is None:
        msg = "datetime must be timezone-aware"
        raise ValueError(msg)
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_id() -> str:
    """Return a new UUID in canonical string form."""
    return str(uuid4())


def is_uuid(value: str) -> bool:
    """Return whether *value* is a canonical UUID string."""
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return str(parsed) == value
