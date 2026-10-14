"""Local policy retriever. Retrieved text is data, never executable instruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campus_ops_agent.tools import load_json


class PolicyRetriever:
    def __init__(self, *, data_dir: Path | None = None) -> None:
        self._documents: list[dict[str, Any]] = load_json("policies.json", data_dir=data_dir)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        terms = {term.lower() for term in query.split() if len(term) > 1}
        if not terms:
            return [dict(item) for item in self._documents]
        matches = [
            item
            for item in self._documents
            if any(term in str(item.get("content", "")).lower() for term in terms)
        ]
        return [dict(item) for item in (matches or self._documents)]


def wrap_retriever(retriever: PolicyRetriever, runtime: Any) -> Any:
    if runtime is None:
        return retriever.retrieve
    return runtime.wrap_retriever(retriever.retrieve, name="campus_docs")


def document_content(document: Any) -> tuple[str, bool, bool]:
    """Return content, mutation provenance, and declared trust."""

    if isinstance(document, dict):
        metadata = document.get("metadata")
        mutated = isinstance(metadata, dict) and "agentqa_mutation" in metadata
        content = document.get("content") or document.get("page_content") or ""
        return str(content), mutated, bool(document.get("trusted", False))
    metadata = getattr(document, "metadata", {})
    mutated = isinstance(metadata, dict) and "agentqa_mutation" in metadata
    return str(getattr(document, "page_content", "")), mutated, False
