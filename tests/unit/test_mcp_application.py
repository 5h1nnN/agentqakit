import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from agentqa.mcp import AgentQAMCPApplication, MCPApplicationError
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.workbench.models import HandoffBundle
from tests.fixtures.artifact_runs import create_run_artifact


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symbolic-link privilege is unavailable")
        raise


def _manifest(path: Path, *, external: bool = False) -> None:
    tool = ""
    if external:
        tool = """
tools:
  - name: charge-card
    description: Charge an external payment card.
    sideEffect: external-write
    risk: high
    idempotent: false
"""
    path.write_text(
        """apiVersion: agentqa.dev/v1alpha2
kind: AgentManifest
metadata:
  name: mcp-agent
runtime:
  language: python
purpose:
  summary: Exercise the MCP boundary.
"""
        + tool,
        encoding="utf-8",
    )


def _resource(value: str) -> dict[str, Any]:
    document = cast(dict[str, Any], json.loads(value))
    assert document["schema_version"] == "agentqa.mcp-resource/v1alpha1"
    assert document["checksum"] == (
        "sha256:" + hashlib.sha256(canonical_json_bytes(document["data"])).hexdigest()
    )
    return document


def test_mcp_resources_are_checksummed_and_trace_is_only_a_summary(tmp_path: Path) -> None:
    _manifest(tmp_path / "agent-manifest.yaml")
    create_run_artifact(tmp_path / "artifacts", "run-mcp", {"case-a": True})
    application = AgentQAMCPApplication(tmp_path)

    manifest = _resource(application.project_manifest_resource())
    schema = _resource(application.schema_resource("agent-test"))
    traces = _resource(application.run_trace_summary_resource("run-mcp"))

    assert manifest["data"]["manifest"]["metadata"]["name"] == "mcp-agent"
    assert schema["data"]["title"] == "AgentTestSpec"
    trace_data = traces["data"]
    assert isinstance(trace_data, dict)
    assert trace_data["traces"][0]["event_count"] == 0
    assert trace_data["traces"][0]["status"] == "complete"
    assert "events" not in trace_data["traces"][0]


def test_missing_dynamic_resource_returns_structured_error(tmp_path: Path) -> None:
    application = AgentQAMCPApplication(tmp_path)

    document = _resource(application.run_summary_resource("unknown-run"))

    assert document["data"]["error"]["code"] == "MCP_RESOURCE_NOT_FOUND"


def test_workspace_rejects_parent_and_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-agentqa.yaml"
    outside.write_text("not: used\n", encoding="utf-8")
    _symlink_or_skip(tmp_path / "escape.yaml", outside)
    application = AgentQAMCPApplication(tmp_path)

    with pytest.raises(MCPApplicationError, match="relative"):
        application.validate_tests("../outside-agentqa.yaml")
    with pytest.raises(MCPApplicationError, match="escapes"):
        application.validate_tests("escape.yaml")


def test_workspace_rejects_output_directory_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-agentqa-artifacts"
    outside.mkdir(exist_ok=True)
    _symlink_or_skip(tmp_path / "artifacts", outside, target_is_directory=True)
    application = AgentQAMCPApplication(tmp_path)

    with pytest.raises(MCPApplicationError, match="escapes"):
        _ = application.artifacts


@pytest.mark.asyncio
async def test_side_effect_manifest_requires_allow_and_approval(tmp_path: Path) -> None:
    _manifest(tmp_path / "agent-manifest.yaml", external=True)
    application = AgentQAMCPApplication(tmp_path)

    with pytest.raises(MCPApplicationError) as disallowed:
        await application.run_tests(tests_path="tests.yaml", target_path="target.yaml")
    assert disallowed.value.code == "MCP_SIDE_EFFECTS_NOT_ALLOWED"

    with pytest.raises(MCPApplicationError) as approval:
        await application.run_tests(
            tests_path="tests.yaml",
            target_path="target.yaml",
            allow_side_effects=True,
        )
    assert approval.value.code == "MCP_APPROVAL_REQUIRED"


def test_compare_tool_writes_resource_backed_report(tmp_path: Path) -> None:
    create_run_artifact(tmp_path / "artifacts", "baseline", {"case-a": True}, cost_usd=0.2)
    create_run_artifact(tmp_path / "artifacts", "candidate", {"case-a": True}, cost_usd=0.1)
    application = AgentQAMCPApplication(tmp_path)

    result = application.compare_runs("baseline", "candidate")
    resource = _resource(application.comparison_resource(result["comparison_id"]))

    assert result["quality_cost_status"] == "QUALITY_PRESERVED_COST_IMPROVED"
    assert resource["data"]["artifact"]["candidate_run_id"] == "candidate"
    report = tmp_path / result["report_path"]
    assert (
        resource["data"]["artifact_checksum"]
        == report.with_name(f"{report.name}.sha256").read_text(encoding="utf-8").strip()
    )


def test_web_handoff_is_available_as_verified_mcp_resource(tmp_path: Path) -> None:
    digest = "sha256:" + "1" * 64
    handoff = HandoffBundle.model_validate(
        {
            "handoff_id": "handoff-1",
            "created_at": datetime(2026, 10, 15, tzinfo=UTC),
            "workflow_id": "workflow-1",
            "project_id": "campus-ops",
            "handoff_resource_uri": ("agentqa://workflows/workflow-1/handoffs/handoff-1"),
            "failure_resource_uri": "agentqa://runs/run-1/failures/trial-1",
            "analysis_resource_uri": "agentqa://analyses/analysis-1",
            "baseline_target_hash": digest,
            "baseline_manifest_hash": digest,
            "baseline_source_snapshot_hash": digest,
            "baseline_runtime_hash": digest,
            "suggested_regression_tests": ["authorization remains enforced"],
            "missing_evidence": [],
            "developer_task": "Fix only the cited authorization regression.",
        }
    )
    destination = (
        tmp_path / ".agentqa/artifacts/workflows/workflow-1/handoffs/handoff-1/handoff-bundle.json"
    )
    destination.parent.mkdir(parents=True)
    payload = canonical_json_bytes(handoff) + b"\n"
    destination.write_bytes(payload)
    destination.with_name(f"{destination.name}.sha256").write_text(
        f"{sha256_digest(payload)}\n", encoding="utf-8"
    )

    resource = _resource(
        AgentQAMCPApplication(tmp_path).handoff_resource("workflow-1", "handoff-1")
    )

    assert resource["data"]["artifact"] == handoff.model_dump(mode="json")
    assert resource["data"]["artifact_checksum"] == sha256_digest(payload)
