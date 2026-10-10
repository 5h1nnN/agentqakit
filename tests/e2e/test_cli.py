from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentqa import __version__
from agentqa.cli import app
from tests.fixtures.artifact_runs import create_run_artifact

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


def test_serve_is_listed_and_rejects_remote_binding_without_opt_in() -> None:
    help_result = runner.invoke(app, ["serve", "--help"])
    remote_result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])

    assert help_result.exit_code == 0
    assert "local AgentQAKit API" in help_result.stdout
    assert remote_result.exit_code == 2
    assert "non-loopback hosts require --allow-remote" in remote_result.stderr


def test_compare_and_gate_commands_write_checksummed_reports(tmp_path: Path) -> None:
    baseline = create_run_artifact(
        tmp_path / "baseline", "baseline-run", {"case": True, "removed": True}
    )
    candidate = create_run_artifact(tmp_path / "candidate", "candidate-run", {"case": False})
    comparison_path = tmp_path / "comparison.json"

    compared = runner.invoke(
        app,
        ["compare", str(baseline), str(candidate), "--output", str(comparison_path)],
    )

    assert compared.exit_code == 0
    assert "1 regressions" in compared.stdout
    assert "1 removed" in compared.stdout
    assert comparison_path.is_file()
    assert (tmp_path / "comparison.json.sha256").is_file()

    gate = tmp_path / "gate.yaml"
    gate.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: QualityGate
metadata:
  name: e2e
rules:
  minPassRate: 1.0
  maxNewRegressions: 0
""",
        encoding="utf-8",
    )
    gate_path = tmp_path / "gate-result.json"
    gated = runner.invoke(
        app,
        [
            "gate",
            "evaluate",
            str(gate),
            "--baseline",
            str(baseline),
            "--result",
            str(candidate),
            "--output",
            str(gate_path),
        ],
    )

    assert gated.exit_code == 1
    assert "FAIL minPassRate" in gated.stdout
    assert "Quality gate: FAIL" in gated.stdout
    assert gate_path.is_file()
    assert (tmp_path / "gate-result.json.sha256").is_file()


def test_gate_returns_exit_two_when_relative_rule_has_no_baseline(tmp_path: Path) -> None:
    candidate = create_run_artifact(tmp_path, "candidate-run", {"case": True})
    gate = tmp_path / "gate.yaml"
    gate.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: QualityGate
metadata:
  name: e2e
rules:
  maxNewRegressions: 0
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["gate", "evaluate", str(gate), "--result", str(candidate)],
    )

    assert result.exit_code == 2
    assert "GATE_BASELINE_REQUIRED" in result.stderr


def test_run_executes_python_target_with_builtin_schema_evaluator(
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
            "--no-trace-receiver",
        ],
    )

    assert result.exit_code == 0
    assert "COMPLETED (1 passed, 0 failed" in result.stdout
    run_directories = list((artifacts / "runs").iterdir())
    assert len(run_directories) == 1
    assert (run_directories[0] / "summary.json").is_file()


def test_run_loads_mutation_and_score_command_recomputes_artifact(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: mutation-graph
type: langgraph
entrypoint: tests.fixtures.langgraph_targets:build_mutation_graph
""",
        encoding="utf-8",
    )
    test = tmp_path / "test.yaml"
    test.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: AgentTest
metadata:
  name: mutation-cli-test
targetRef: mutation-graph
scenario:
  input:
    message: hello
mutation: lookup-exception
expect:
  output:
    requireNonEmpty: true
evaluators:
  - type: schema
""",
        encoding="utf-8",
    )
    mutation = tmp_path / "mutation.yaml"
    mutation.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: Mutation
metadata:
  name: lookup-exception
target:
  kind: tool
  name: campus.lookup
operation:
  type: tool_exception
""",
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"

    run_result = runner.invoke(
        app,
        [
            "test",
            "run",
            str(test),
            "--target",
            str(target),
            "--mutations",
            str(mutation),
            "--artifacts",
            str(artifacts),
            "--no-trace-receiver",
        ],
    )

    assert run_result.exit_code == 1
    assert "Mutation score: 100.0%" in run_result.stdout
    run_directory = next((artifacts / "runs").iterdir())
    assert (run_directory / "mutation-score.json").is_file()

    score_result = runner.invoke(app, ["mutation", "score", str(run_directory)])

    assert score_result.exit_code == 0
    assert "Mutation score: 100.0%" in score_result.stdout
    assert "Killed: 1" in score_result.stdout

    failure_bundle = next((run_directory / "failures").glob("failure-*.zip"))
    replay_artifacts = tmp_path / "replay-artifacts"
    replay_result = runner.invoke(
        app,
        [
            "replay",
            str(failure_bundle),
            "--artifacts",
            str(replay_artifacts),
        ],
    )

    assert replay_result.exit_code == 0
    assert "Reproduced failure" in replay_result.stdout
    assert len(list((replay_artifacts / "runs").iterdir())) == 1
