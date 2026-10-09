"""Small LangGraph-compatible fake graphs with no optional dependency."""

from collections.abc import AsyncIterator
from typing import Any

_REPLAY_TOOL_STATE = {"available": True, "calls": 0}


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


class MutationEventGraph:
    def __init__(self, tool: Any, *, stringify_result: bool = True) -> None:
        self._tool = tool
        self._stringify_result = stringify_result

    async def astream_events(
        self,
        value: dict[str, Any],
        *,
        config: dict[str, Any],
        version: str,
    ) -> AsyncIterator[dict[str, Any]]:
        del config
        assert version == "v2"
        yield {
            "event": "on_chain_start",
            "name": "mutation-fixture-agent",
            "run_id": "mutation-agent-span",
            "parent_ids": [],
            "data": {"input": value},
        }
        result = self._tool(query="campus calendar")
        content = str(result) if self._stringify_result else result
        output = {"messages": [{"role": "assistant", "content": content}]}
        yield {
            "event": "on_chain_end",
            "name": "mutation-fixture-agent",
            "run_id": "mutation-agent-span",
            "parent_ids": [],
            "data": {"output": output},
        }


def build_graph(agentqa_runtime: Any = None) -> EventGraph:
    assert agentqa_runtime is not None
    return EventGraph()


def build_mutation_graph(agentqa_runtime: Any = None) -> MutationEventGraph:
    assert agentqa_runtime is not None

    def campus_lookup(query: str) -> dict[str, Any]:
        return {"query": query, "answer": "semester starts Monday"}

    tool = agentqa_runtime.wrap_tool(campus_lookup, name="campus.lookup")
    return MutationEventGraph(tool)


def configure_replay_tool(*, available: bool, reset_calls: bool = False) -> None:
    _REPLAY_TOOL_STATE["available"] = available
    if reset_calls:
        _REPLAY_TOOL_STATE["calls"] = 0


def replay_tool_calls() -> int:
    return int(_REPLAY_TOOL_STATE["calls"])


def build_replay_graph(agentqa_runtime: Any = None) -> MutationEventGraph:
    assert agentqa_runtime is not None

    def campus_lookup(query: str) -> dict[str, Any]:
        if not _REPLAY_TOOL_STATE["available"]:
            raise RuntimeError("the real tool server is offline")
        _REPLAY_TOOL_STATE["calls"] += 1
        return {"query": query, "answer": "recorded calendar"}

    tool = agentqa_runtime.wrap_tool(campus_lookup, name="campus.lookup")
    return MutationEventGraph(tool, stringify_result=False)


compiled_graph = InvokeGraph()
