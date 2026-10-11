"""Small decorated Python agent used by the Web mutation demo."""

from typing import Any

import agentqa


@agentqa.tool(name="query_course")
def query_course(course_id: str) -> dict[str, str]:
    """Return one deterministic synthetic course record."""
    return {
        "courseId": course_id,
        "course": "人工智能导论",
        "classTime": "周一 09:00",
        "room": "A101",
    }


@agentqa.agent(name="sdk-course-agent")
def run_agent(payload: dict[str, Any]) -> dict[str, str]:
    """Call the decorated tool and expose its result as the agent answer."""
    course_id = str(payload.get("input", {}).get("course_id", "AI101"))
    course = query_course(course_id)
    return {"status": "ok", "answer": "查询成功", **course}
