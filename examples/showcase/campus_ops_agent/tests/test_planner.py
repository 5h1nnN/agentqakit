import pytest

from campus_ops_agent.planner import _validated_action


def test_model_action_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="outside"):
        _validated_action({"name": "query_course", "shell_command": "whoami"})


def test_model_action_rejects_role_escalation_shape() -> None:
    with pytest.raises(ValueError, match="role"):
        _validated_action({"name": "query_score", "role": "admin"})
