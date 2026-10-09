from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentqa import __version__
from agentqa.cli import app

runner = CliRunner()
pytestmark = pytest.mark.e2e


def test_help_lists_top_level_usage() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Fault-injection and regression testing for AI agents." in result.stdout
    assert "--version" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"AgentQAKit {__version__}"


def test_run_executes_python_target_and_fails_closed_without_t6_evaluator(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: python-fixture
type: python
entrypoint: tests.fixtures.python_targets:echo
""",
        encoding="utf-8",
    )
    test = tmp_path / "test.yaml"
    test.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: AgentTest
metadata:
  name: cli-test
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
    artifacts = tmp_path / "artifacts"

    result = runner.invoke(
        app,
        [
            "test",
            "run",
            str(test),
            "--target",
            str(target),
            "--artifacts",
            str(artifacts),
        ],
    )

    assert result.exit_code == 1
    assert "COMPLETED (0 passed, 1 failed" in result.stdout
    run_directories = list((artifacts / "runs").iterdir())
    assert len(run_directories) == 1
    assert (run_directories[0] / "summary.json").is_file()
