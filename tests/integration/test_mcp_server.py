import json
from pathlib import Path

import pytest
from mcp import Client
from mcp.types import TextResourceContents

from agentqa.comparison import load_run_artifact
from agentqa.mcp.server import create_mcp_server


def _workspace_specs(root: Path) -> None:
    (root / "target.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: python-fixture
type: python
entrypoint: tests.fixtures.python_targets:echo
isolation: subprocess
workerMode: perInvocation
timeoutMs: 10000
""",
        encoding="utf-8",
    )
    (root / "test.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: AgentTest
metadata:
  name: mcp-test
targetRef: python-fixture
scenario:
  input:
    message: hello
expect:
  output:
    requireNonEmpty: true
evaluators:
  - type: schema
""",
        encoding="utf-8",
    )
    (root / "agent-manifest.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha2
kind: AgentManifest
metadata:
  name: python-fixture
runtime:
  language: python
  framework: custom
  entrypoint: tests.fixtures.python_targets:echo
purpose:
  summary: Echo a bounded MCP test input.
""",
        encoding="utf-8",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_official_in_memory_client_discovers_and_runs_shared_runner(tmp_path: Path) -> None:
    _workspace_specs(tmp_path)
    server = create_mcp_server(tmp_path)

    async with Client(server, raise_exceptions=True) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        resources = {str(item.uri) for item in (await client.list_resources()).resources}
        templates = {
            item.uri_template
            for item in (await client.list_resource_templates()).resource_templates
        }
        validated = await client.call_tool(
            "validate_manifest", {"manifest_path": "agent-manifest.yaml"}
        )
        executed = await client.call_tool(
            "run_tests",
            {
                "tests_path": "test.yaml",
                "target_path": "target.yaml",
                "manifest_path": "agent-manifest.yaml",
            },
        )

        assert tools == {
            "validate_manifest",
            "generate_tests",
            "validate_tests",
            "run_tests",
            "run_mutations",
            "get_run_result",
            "analyze_failure",
            "compare_runs",
            "evaluate_gate",
        }
        assert resources == {
            "agentqa://project/manifest",
            "agentqa://schemas/agent-test",
            "agentqa://schemas/agent-manifest",
        }
        assert templates == {
            "agentqa://runs/{run_id}/summary",
            "agentqa://runs/{run_id}/trace-summary",
            "agentqa://runs/{run_id}/failures/{failure_id}",
            "agentqa://comparisons/{comparison_id}",
            "agentqa://analyses/{analysis_id}",
            "agentqa://workflows/{workflow_id}/handoffs/{handoff_id}",
        }
        assert validated.structured_content is not None
        assert validated.structured_content["valid"] is True
        assert executed.structured_content is not None
        run_id = executed.structured_content["run_id"]
        assert executed.structured_content["status"] == "COMPLETED"
        assert executed.structured_content["exit_code"] == 0

        summary = await client.read_resource(f"agentqa://runs/{run_id}/summary")
        content = summary.contents[0]
        assert isinstance(content, TextResourceContents)
        resource = json.loads(content.text)
        assert resource["data"]["run_id"] == run_id

    artifact = load_run_artifact(tmp_path / "artifacts" / "runs" / run_id)
    assert executed.structured_content["summary"]["total"] == artifact.summary.total
    assert executed.structured_content["summary"]["case_count"] == len(artifact.summary.cases)
    assert "cases" not in executed.structured_content["summary"]
    assert artifact.results[0].test_case_id == "mcp-test"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_tool_rejects_workspace_escape_as_safe_error(tmp_path: Path) -> None:
    server = create_mcp_server(tmp_path)

    async with Client(server) as client:
        result = await client.call_tool("validate_tests", {"tests_path": "../escape.yaml"})

    assert result.is_error is True
    assert "MCP_PATH_ERROR" in result.content[0].text  # type: ignore[union-attr]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_run_rejects_unknown_target_reference(tmp_path: Path) -> None:
    _workspace_specs(tmp_path)
    test_path = tmp_path / "test.yaml"
    test_path.write_text(
        test_path.read_text(encoding="utf-8").replace(
            "targetRef: python-fixture",
            "targetRef: unknown-target",
        ),
        encoding="utf-8",
    )
    server = create_mcp_server(tmp_path)

    async with Client(server) as client:
        result = await client.call_tool(
            "run_tests",
            {
                "tests_path": "test.yaml",
                "target_path": "target.yaml",
                "manifest_path": "agent-manifest.yaml",
            },
        )

    assert result.is_error is True
    assert "RUN_SERVICE_TARGET_MISMATCH" in result.content[0].text  # type: ignore[union-attr]
