from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest

from agentqa.conventions import is_uuid, new_id, to_utc_iso8601, utc_now


def test_utc_now_is_timezone_aware_and_utc() -> None:
    value = utc_now()

    assert value.tzinfo is UTC
    assert value.utcoffset() == timedelta(0)


def test_to_utc_iso8601_normalizes_offset_and_uses_z_suffix() -> None:
    value = datetime(2026, 10, 9, 16, 30, tzinfo=timezone(timedelta(hours=8)))

    assert to_utc_iso8601(value) == "2026-10-09T08:30:00.000000Z"


def test_to_utc_iso8601_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        to_utc_iso8601(datetime(2026, 10, 9))


def test_new_id_returns_canonical_uuid() -> None:
    value = new_id()

    assert str(UUID(value)) == value
    assert is_uuid(value)
    assert not is_uuid(value.upper())
    assert not is_uuid("not-a-uuid")
