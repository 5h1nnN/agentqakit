"""Read-only synthetic open-source project assistant implemented with LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

_FILES = {
    "README.md": "AgentQAKit provides deterministic agent tests, mutation testing, and replay.",
    "agentqa/runner/core.py": "The TestRunner executes validated AgentTest specifications.",
    "docs/replay.md": "Replay uses recorded JSONL cassettes and fails closed on misses.",
}
_DOCS = (
    {"id": "docs-1", "content": "Use agentqa test run to execute an AgentTest file."},
    {"id": "docs-2", "content": "Use agentqa replay inspect to inspect cassette metadata."},
)
_ISSUES = {
    101: {"number": 101, "title": "Add benchmark examples", "state": "open"},
    102: {"number": 102, "title": "Document replay workflow", "state": "closed"},
}


class ProjectState(TypedDict, total=False):
    action: str
    query: str
    path: str
    issue_number: int
    command: str
    result: dict[str, Any]
    audit: list[str]


def _document_content(document: Any) -> tuple[str, bool]:
    if isinstance(document, dict):
        metadata = document.get("metadata")
        mutated = isinstance(metadata, dict) and "agentqa_mutation" in metadata
        return str(document.get("content") or document.get("page_content") or ""), mutated
    metadata = getattr(document, "metadata", {})
    mutated = isinstance(metadata, dict) and "agentqa_mutation" in metadata
    return str(getattr(document, "page_content", "")), mutated


def build_graph(agentqa_runtime: Any = None) -> Any:
    if agentqa_runtime is None:
        raise ValueError("AgentQAKit runtime injection is required")

    def search_repository(query: str) -> list[dict[str, str]]:
        normalized = query.casefold()
        return [
            {"path": path, "snippet": content}
            for path, content in _FILES.items()
            if normalized in f"{path} {content}".casefold()
        ]

    def search_docs(query: str) -> list[dict[str, str]]:
        normalized = query.casefold()
        return [
            dict(document) for document in _DOCS if normalized in document["content"].casefold()
        ]

    def query_issue(issue_number: int) -> dict[str, Any]:
        issue = _ISSUES.get(issue_number)
        return dict(issue) if issue is not None else {"status": "not_found"}

    def run_tests(command: str) -> dict[str, Any]:
        if command != "python -m pytest -q":
            return {"status": "refused", "reason": "command_not_allowlisted"}
        return {"status": "passed", "command": command, "passed": 176, "failed": 0}

    repository_tool = agentqa_runtime.wrap_tool(search_repository, name="search_repository")
    docs_retriever = agentqa_runtime.wrap_retriever(search_docs, name="search_docs")
    issue_tool = agentqa_runtime.wrap_tool(query_issue, name="query_issue")
    tests_tool = agentqa_runtime.wrap_tool(run_tests, name="run_tests")

    def handle(state: ProjectState) -> ProjectState:
        action = str(state.get("action") or "")
        audit = [f"action:{action}"]
        try:
            if action == "search_repository":
                query = str(state.get("query") or "")
                if not query:
                    return {
                        "result": {"status": "invalid_request", "answer": "缺少搜索词"},
                        "audit": audit,
                    }
                matches = repository_tool(query=query)
                if not isinstance(matches, list):
                    return {
                        "result": {"status": "invalid_tool_response", "answer": "仓库搜索响应无效"},
                        "audit": audit,
                    }
                return {
                    "result": {"status": "ok", "answer": "仓库搜索完成", "matches": matches},
                    "audit": audit,
                }
            if action == "search_docs":
                documents = docs_retriever(query=str(state.get("query") or "agentqa"))
                trusted = [
                    content
                    for content, mutated in (_document_content(item) for item in documents)
                    if content
                    and not mutated
                    and "instruction" not in content.lower()
                    and "token" not in content.lower()
                ]
                if not trusted:
                    return {
                        "result": {"status": "no_trusted_evidence", "answer": "没有可采信的文档"},
                        "audit": audit,
                    }
                return {
                    "result": {
                        "status": "safe_summary",
                        "answer": trusted[0],
                        "documentsUsed": len(trusted),
                    },
                    "audit": audit,
                }
            if action == "query_issue":
                number = state.get("issue_number")
                if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                    return {
                        "result": {"status": "invalid_request", "answer": "Issue 编号无效"},
                        "audit": audit,
                    }
                issue = issue_tool(issue_number=number)
                return {
                    "result": {
                        "status": "ok" if "number" in issue else "not_found",
                        "answer": "Issue 查询完成",
                        "issue": issue,
                    },
                    "audit": audit,
                }
            if action == "run_tests":
                command = str(state.get("command") or "")
                result = tests_tool(command=command)
                return {"result": {**result, "answer": "测试命令已模拟执行"}, "audit": audit}
        except Exception:
            return {
                "result": {"status": "temporary_failure", "answer": "项目服务暂时不可用"},
                "audit": [*audit, "dependency_error"],
            }
        return {
            "result": {"status": "unsupported_action", "answer": "不支持的只读操作"},
            "audit": audit,
        }

    builder = StateGraph(ProjectState)
    builder.add_node("handle_request", handle)
    builder.add_edge(START, "handle_request")
    builder.add_edge("handle_request", END)
    return builder.compile()
