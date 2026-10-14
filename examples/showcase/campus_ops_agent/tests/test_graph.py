import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="langgraph extra not installed",
)


@pytest.mark.asyncio
async def test_fixed_graph_runs_standalone() -> None:
    from campus_ops_agent.graph import build_fixed

    result = await build_fixed().ainvoke({"action": "query_course", "course_id": "AI101"})
    assert result["result"]["status"] == "ok"
    assert result["result"]["courseId"] == "AI101"
