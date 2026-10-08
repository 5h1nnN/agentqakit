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
