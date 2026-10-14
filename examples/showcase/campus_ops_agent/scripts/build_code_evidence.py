#!/usr/bin/env python3
"""Build bounded, hash-verified CodeEvidence for the T16 code-aware condition."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = (
    (
        "src/campus_ops_agent/service.py",
        "CampusOpsService.invoke",
        "Authorization order around private tools.",
    ),
    (
        "src/campus_ops_agent/service.py",
        "CampusOpsService._course",
        "Bounded retry loop and tool response validation.",
    ),
    (
        "src/campus_ops_agent/service.py",
        "CampusOpsService._register",
        "Confirmation-protected write call and idempotency key flow.",
    ),
    (
        "src/campus_ops_agent/service.py",
        "CampusOpsService._policy",
        "Retrieved-content trust boundary.",
    ),
    (
        "src/campus_ops_agent/policies.py",
        "validation_error",
        "Confirmation and required-parameter policy.",
    ),
    ("src/campus_ops_agent/policies.py", "idempotency_key", "Stable idempotency-key derivation."),
    (
        "src/campus_ops_agent/tools.py",
        "CampusTools.register_event",
        "Write deduplication and duplicate defect branch.",
    ),
)


def _find_node(tree: ast.AST, qualified_name: str) -> ast.AST:
    parts = qualified_name.split(".")
    nodes: list[ast.AST] = list(ast.iter_child_nodes(tree))
    for part in parts:
        match = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if match is None:
            raise ValueError(f"cannot find {qualified_name}")
        nodes = list(ast.iter_child_nodes(match))
    return match


def build() -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for relative_path, symbol, reason in SELECTIONS:
        path = ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        node = _find_node(ast.parse(source), symbol)
        lines = source.splitlines()
        start = node.lineno
        end = node.end_lineno or node.lineno
        content = "\n".join(lines[start - 1 : end])
        digest = hashlib.sha256(content.encode()).hexdigest()
        evidence.append(
            {
                "path": relative_path,
                "startLine": start,
                "endLine": end,
                "contentHash": f"sha256:{digest}",
                "reason": reason,
                "content": content,
            }
        )
    return evidence


def main() -> None:
    output_dir = ROOT / "agentqa" / "code-evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = build()
    (output_dir / "t16-code-evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    goal = yaml.safe_load((ROOT / "agentqa" / "generation-goal.yaml").read_text(encoding="utf-8"))
    goal["metadata"]["name"] = "campus-ops-t16-code-generation"
    goal["codeEvidence"] = evidence
    (ROOT / "agentqa" / "generation-goal-code.yaml").write_text(
        yaml.safe_dump(goal, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {len(evidence)} excerpts")


if __name__ == "__main__":
    main()
