"""Small LangGraph-compatible fake graphs with no optional dependency."""

from collections.abc import AsyncIterator
from typing import Any


class EventGraph:
    async def astream_events(
        self,
        value: dict[str, Any],
        *,
        config: dict[str, Any],
        version: str,
    ) -> AsyncIterator[dict[str, Any]]:
        assert version == "v2"
        assert "agentqa" in config["tags"]
        yield {
            "event": "on_chain_start",
            "name": "fixture-agent",
            "run_id": "agent-span",
            "parent_ids": [],
            "data": {"input": value},
        }
        yield {
            "event": "on_chat_model_start",
            "name": "fixture-model",
            "run_id": "llm-span",
            "parent_ids": ["agent-span"],
            "data": {"input": value.get("messages")},
        }
        yield {
            "event": "on_chat_model_end",
            "name": "fixture-model",
            "run_id": "llm-span",
            "parent_ids": ["agent-span"],
            "data": {"output": {"content": "done"}},
        }
        yield {
            "event": "on_tool_start",
            "name": "fixture-tool",
            "run_id": "tool-span",
            "parent_ids": ["agent-span"],
            "data": {"input": {"query": "test"}},
        }
        yield {
            "event": "on_tool_end",
            "name": "fixture-tool",
            "run_id": "tool-span",
            "parent_ids": ["agent-span"],
            "data": {"output": {"answer": "tool-result"}},
        }
        output = {"messages": [{"role": "assistant", "content": "done"}], "safe": 1}
        yield {
            "event": "on_chain_end",
            "name": "fixture-agent",
            "run_id": "agent-span",
            "parent_ids": [],
            "data": {"output": output},
        }


class InvokeGraph:
    async def ainvoke(self, value: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        assert config["metadata"]["agentqa_run_id"]
        return {"answer": value["question"]}


def build_graph(agentqa_runtime: Any = None) -> EventGraph:
    assert agentqa_runtime is not None
    return EventGraph()


compiled_graph = InvokeGraph()
