"""A real optional LangGraph target used only by integration tests."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class _State(TypedDict):
    messages: list[dict[str, Any]]


def _answer(state: _State) -> _State:
    return {
        "messages": [
            *state["messages"],
            {"role": "assistant", "content": "real-graph-done"},
        ]
    }


def build_real_graph(agentqa_runtime: Any = None) -> Any:
    assert agentqa_runtime is not None
    builder = StateGraph(_State)
    builder.add_node("answer", _answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def build_real_mutation_graph(agentqa_runtime: Any = None) -> Any:
    assert agentqa_runtime is not None

    def campus_lookup(query: str) -> dict[str, str]:
        return {"query": query, "answer": "semester starts Monday"}

    wrapped_lookup = agentqa_runtime.wrap_tool(campus_lookup, name="campus.lookup")

    def answer_with_tool(state: _State) -> _State:
        result = wrapped_lookup(query="campus calendar")
        return {
            "messages": [
                *state["messages"],
                {"role": "assistant", "content": result["answer"]},
            ]
        }

    builder = StateGraph(_State)
    builder.add_node("answer", answer_with_tool)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()
