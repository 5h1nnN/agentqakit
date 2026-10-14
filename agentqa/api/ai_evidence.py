"""Read-only, checksummed AI generation and analysis summaries for the Web console."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agentqa.models import AnalysisReport, GenerationReport
from agentqa.specs.hashing import sha256_digest

_MAX_REPORT_BYTES = 4 * 1024 * 1024
_MAX_REPORTS = 100


def _candidates(workspace: Path, patterns: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path.resolve() for path in workspace.glob(pattern) if path.is_file())
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)[:_MAX_REPORTS]


def _verified(path: Path, workspace: Path) -> dict[str, Any] | None:
    try:
        path.relative_to(workspace)
        payload = path.read_bytes()
        expected = path.with_name(f"{path.name}.sha256").read_text(encoding="utf-8").strip()
        if len(payload) > _MAX_REPORT_BYTES or expected != sha256_digest(payload):
            return None
        value = json.loads(payload)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def _llm_call(value: Any) -> dict[str, Any]:
    return {
        "use_case": value.use_case,
        "provider": value.provider,
        "model": value.model,
        "status": value.status,
        "attempts": value.attempts,
        "token_usage": value.token_usage,
        "latency_ms": value.latency_ms,
        "evidence_hash": value.evidence_hash,
    }


def list_ai_evidence(workspace_root: Path) -> dict[str, Any]:
    """Return bounded summaries from known local artifact layouts."""
    workspace = workspace_root.resolve()
    generations: list[dict[str, Any]] = []
    generation_paths = _candidates(
        workspace,
        (
            ".agentqa/mcp/drafts/*/tests/generation-report.json",
            "benchmark/results/*/arms/*/generated/generation-report.json",
            "benchmark/reference-results/*/arms/*/generated/generation-report.json",
        ),
    )
    for path in generation_paths:
        value = _verified(path, workspace)
        if value is None:
            continue
        try:
            generation_report = GenerationReport.model_validate(value)
        except ValueError:
            continue
        generations.append(
            {
                "generation_id": generation_report.generation_id,
                "path": path.relative_to(workspace).as_posix(),
                "generated_at": generation_report.generated_at,
                "manifest_hash": generation_report.manifest_hash,
                "plan_hash": generation_report.plan_hash,
                "context_hashes": generation_report.code_evidence_hashes,
                "coverage_gaps": len(generation_report.gaps),
                "cases": [
                    {
                        "case_id": case.case_id,
                        "risk": case.risk,
                        "review_status": case.review_status,
                        "source_refs": case.source_refs,
                    }
                    for case in generation_report.cases
                ],
                "llm_calls": [_llm_call(call) for call in generation_report.llm_calls],
            }
        )

    analyses: list[dict[str, Any]] = []
    analysis_paths = _candidates(
        workspace,
        (
            ".agentqa/mcp/analyses/*.json",
            "benchmark/results/*/standalone/analysis/*.json",
            "benchmark/reference-results/*/standalone/analysis/*.json",
        ),
    )
    for path in analysis_paths:
        value = _verified(path, workspace)
        if value is None:
            continue
        try:
            analysis_report = AnalysisReport.model_validate(value)
        except ValueError:
            continue
        analyses.append(
            {
                "analysis_id": analysis_report.analysis_id,
                "path": path.relative_to(workspace).as_posix(),
                "generated_at": analysis_report.generated_at,
                "run_id": analysis_report.run_id,
                "test_case_id": analysis_report.test_case_id,
                "trial_id": analysis_report.trial_id,
                "evidence_hash": analysis_report.evidence_hash,
                "missing_evidence": analysis_report.missing_evidence,
                "root_causes": [
                    item.model_dump(mode="json") for item in analysis_report.root_causes
                ],
                "llm_call": _llm_call(analysis_report.llm_call),
            }
        )
    return {"generations": generations, "analyses": analyses}


__all__ = ["list_ai_evidence"]
