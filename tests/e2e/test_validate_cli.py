from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentqa.cli import app

runner = CliRunner()
pytestmark = pytest.mark.e2e


def test_validate_example_specs() -> None:
    examples = Path(__file__).resolve().parents[2] / "examples" / "specs"

    result = runner.invoke(app, ["test", "validate", str(examples)])

    assert result.exit_code == 0
    assert "Validated 20 specification(s)." in result.stdout


def test_validate_reports_safe_error_and_nonzero_exit(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: AgentTest
metadata: {}
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["test", "validate", str(invalid)])

    assert result.exit_code == 1
    assert "metadata.name" in result.stderr
    assert "Validation failed" in result.stderr
