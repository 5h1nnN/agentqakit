from campus_ops_agent.tools import CampusTools


def test_course_lookup() -> None:
    result = CampusTools().query_course("AI101")
    assert result["course"] == "人工智能导论"


def test_registration_is_idempotent() -> None:
    tools = CampusTools()
    first = tools.register_event("student-001", "event-robotics", "same-key")
    second = tools.register_event("student-001", "event-robotics", "same-key")
    assert first["registrationId"] == second["registrationId"]
    assert second["status"] == "already_registered"


def test_bug_variant_creates_duplicate_registration() -> None:
    tools = CampusTools(enforce_idempotency=False)
    first = tools.register_event("student-001", "event-robotics", "same-key")
    second = tools.register_event("student-001", "event-robotics", "same-key")
    assert first["registrationId"] != second["registrationId"]
