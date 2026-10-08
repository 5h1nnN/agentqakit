from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agentqa.models import ErrorInfo, GateResult, TargetResult


def test_runtime_datetime_serializes_as_utc_z() -> None:
    started = datetime(2026, 10, 9, 8, 30, tzinfo=UTC)
    result = TargetResult(
        status="OK",
        output={"answer": "ok"},
        started_at=started,
        ended_at=started + timedelta(seconds=1),
    )

    serialized = result.model_dump(mode="json")

    assert serialized["started_at"] == "2026-10-09T08:30:00.000000Z"
    assert serialized["ended_at"] == "2026-10-09T08:30:01.000000Z"


def test_failed_target_result_requires_error_details() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="requires an error"):
        TargetResult(status="ERROR", started_at=now, ended_at=now)

    error = ErrorInfo(code="TARGET_RUNTIME_ERROR", message="failed")
    result = TargetResult(status="ERROR", error=error, started_at=now, ended_at=now)
    assert result.error == error


def test_gate_exit_code_must_match_decision() -> None:
    with pytest.raises(ValidationError, match="exit_code"):
        GateResult(passed=False, exit_code=0)
