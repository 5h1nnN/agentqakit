import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from agentqa.mcp import AgentQAMCPApplication, MCPApplicationError
from agentqa.specs.hashing import canonical_json_bytes
from tests.fixtures.artifact_runs import create_run_artifact


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
    (tmp_path / "escape.yaml").symlink_to(outside)
    application = AgentQAMCPApplication(tmp_path)

    with pytest.raises(MCPApplicationError, match="relative"):
        application.validate_tests("../outside-agentqa.yaml")
    with pytest.raises(MCPApplicationError, match="escapes"):
        application.validate_tests("escape.yaml")


def test_workspace_rejects_output_directory_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-agentqa-artifacts"
    outside.mkdir(exist_ok=True)
    (tmp_path / "artifacts").symlink_to(outside, target_is_directory=True)
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
    assert resource["data"]["candidate_run_id"] == "candidate"
