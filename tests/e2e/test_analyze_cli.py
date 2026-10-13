from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from agentqa.cli import app
from agentqa.llm import StructuredRequest, StructuredResponse
from tests.fixtures.artifact_runs import create_run_artifact

runner = CliRunner()
pytestmark = pytest.mark.e2e


class _FakeProvider:
    name = "fake"
    model = "fake-analysis"
    parameters: Mapping[str, JsonValue] = {"temperature": 0.0}

    async def complete(self, request: StructuredRequest) -> StructuredResponse:
        del request
        return StructuredResponse(
            content={
                "observed_facts": [
                    {
                        "statement": "The normalized target result is part of a failed trial.",
                        "evidence_ids": ["target-result"],
                    }
                ],
                "earliest_divergence": None,
                "root_causes": [
                    {
                        "id": "output-mismatch",
                        "hypothesis": "The output did not satisfy the declared expectation.",
                        "confidence": 0.6,
                        "evidence_ids": ["target-result"],
                        "counter_evidence_ids": [],
                        "reasoning": "The immutable trial outcome is failed.",
                    }
                ],
                "additional_evidence_needed": [],
                "regression_tests": [
                    {
                        "title": "Repeat failed input",
                        "rationale": "Preserve the observed failure as a regression case.",
                        "expected_behavior": "The declared expectation passes.",
                        "evidence_ids": ["scenario", "target-result"],
                    }
                ],
                "repair_directions": [],
            },
            metadata={},
        )


def test_analyze_failed_trial_writes_checksummed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = create_run_artifact(tmp_path / "artifacts", "failed-run", {"failed-case": False})
    output = tmp_path / "analysis" / "report.json"
    provider = _FakeProvider()
    monkeypatch.setattr("agentqa.cli.analyze._provider", lambda *_args: provider)

    result = runner.invoke(
        app,
        [
            "analyze",
            "failure",
            "--run",
            str(run),
            "--trial-id",
            "trial-failed-case",
            "--output",
            str(output),
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1 root-cause candidate" in result.stdout
    assert output.is_file()
    assert output.with_name("report.json.sha256").is_file()
    assert not (run / "analysis-report.json").exists()
