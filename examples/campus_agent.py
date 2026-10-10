"""Small deterministic agent used by local examples and CI."""

from typing import Any


def run_baseline_agent(payload: dict[str, Any]) -> dict[str, str]:
    """Return the frozen reference behavior used by the CI baseline run."""
    del payload
    return {
        "answer": "查询成功",
        "course": "人工智能导论",
        "classTime": "周一 09:00",
        "room": "A101",
    }


def run_agent(payload: dict[str, Any]) -> dict[str, str]:
    """Return the candidate behavior under test."""
    message = str(payload.get("input", {}).get("message", ""))
    return {
        "answer": "查询成功" if message else "缺少查询内容",
        "course": "人工智能导论",
        "classTime": "周一 09:00",
        "room": "A101",
    }
