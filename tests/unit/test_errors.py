from agentqa.errors import AgentQAError


def test_agentqa_error_has_safe_structured_representation() -> None:
    error = AgentQAError(
        "target timed out",
        code="TARGET_TIMEOUT",
        details={"timeout_seconds": 2},
    )

    assert str(error) == "target timed out"
    assert error.to_dict() == {
        "code": "TARGET_TIMEOUT",
        "message": "target timed out",
        "details": {"timeout_seconds": 2},
    }


def test_agentqa_error_does_not_share_default_details() -> None:
    first = AgentQAError("first")
    second = AgentQAError("second")

    first.details["changed"] = True

    assert second.details == {}
