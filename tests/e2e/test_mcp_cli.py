import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from agentqa.comparison import load_run_artifact

pytestmark = pytest.mark.e2e


def _workspace_specs(root: Path) -> None:
    (root / "agent-manifest.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha2
kind: AgentManifest
metadata:
  name: mcp-stdio
runtime:
  language: python
purpose:
  summary: Validate the real stdio protocol boundary.
""",
        encoding="utf-8",
    )
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
  name: mcp-stdio-test
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


@pytest.mark.asyncio
async def test_mcp_serve_uses_clean_stdio_protocol(tmp_path: Path) -> None:
    _workspace_specs(tmp_path)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AGENTQA_LLM_", "AGENTQA_JUDGE_"))
    }
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "agentqa",
            "mcp",
            "serve",
            "--workspace",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
    )

    async with Client(parameters, raise_exceptions=True) as client:
        tools = await client.list_tools()
        validated = await client.call_tool("validate_manifest", {})
        executed = await client.call_tool(
            "run_tests",
            {"tests_path": "test.yaml", "target_path": "target.yaml"},
        )

    assert len(tools.tools) == 9
    assert validated.structured_content is not None
    assert validated.structured_content["valid"] is True
    assert executed.structured_content is not None
    mcp_run_id = executed.structured_content["run_id"]
    mcp_run = load_run_artifact(tmp_path / "artifacts" / "runs" / mcp_run_id)

    cli_artifacts = tmp_path / "cli-artifacts"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "agentqa",
        "test",
        "run",
        str(tmp_path / "test.yaml"),
        "--target",
        str(tmp_path / "target.yaml"),
        "--manifest",
        str(tmp_path / "agent-manifest.yaml"),
        "--artifacts",
        str(cli_artifacts),
        "--no-trace-receiver",
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    assert process.returncode == 0, (stdout + stderr).decode()
    cli_directory = next((cli_artifacts / "runs").iterdir())
    cli_run = load_run_artifact(cli_directory)

    assert executed.structured_content["status"] == "COMPLETED"
    assert executed.structured_content["exit_code"] == process.returncode == 0
    assert (
        mcp_run.summary.total,
        mcp_run.summary.passed,
        mcp_run.summary.failed,
        mcp_run.summary.total_trials,
    ) == (
        cli_run.summary.total,
        cli_run.summary.passed,
        cli_run.summary.failed,
        cli_run.summary.total_trials,
    )
    assert mcp_run.results[0].test_case_id == cli_run.results[0].test_case_id
    assert {path.name for path in mcp_run.root.iterdir()} == {
        path.name for path in cli_run.root.iterdir()
    }


def test_base_cli_imports_without_mcp_extra_and_scopes_install_error() -> None:
    script = """
import json
import sys

class BlockMCP:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mcp" or fullname.startswith("mcp."):
            raise ModuleNotFoundError("blocked optional dependency", name=fullname)
        return None

sys.meta_path.insert(0, BlockMCP())
from typer.testing import CliRunner
from agentqa.cli import app

runner = CliRunner()
version = runner.invoke(app, ["--version"])
serve = runner.invoke(app, ["mcp", "serve", "--workspace", "."])
print(json.dumps({
    "version_code": version.exit_code,
    "version_output": version.stdout,
    "serve_code": serve.exit_code,
    "serve_error": serve.stderr,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["version_code"] == 0
    assert result["version_output"].startswith("AgentQAKit ")
    assert result["serve_code"] == 2
    assert 'pip install "agentqakit[mcp]"' in result["serve_error"]
