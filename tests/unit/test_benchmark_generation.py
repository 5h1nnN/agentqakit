import json
from pathlib import Path

import pytest

from agentqa.models import ErrorInfo, RunResult, RunSummary
from scripts.generate_benchmark import CATEGORY_TARGETS, generate
from scripts.run_benchmark import BenchmarkExecutionError, _require_completed


def test_benchmark_generation_is_deterministic_and_versioned(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    checked_in = Path(__file__).resolve().parents[2] / "benchmark"

    first_manifest = generate(first)
    second_manifest = generate(second)

    assert first_manifest == second_manifest
    assert first_manifest["caseCount"] == 110
    assert first_manifest["goldCaseCount"] == 20
    assert first_manifest["mutationCount"] == 35
    assert first_manifest["categoryCounts"] == CATEGORY_TARGETS
    assert first_manifest["hash"].startswith("sha256:")
    assert json.loads((first / "manifest.json").read_text(encoding="utf-8")) == first_manifest
    for relative in (*first_manifest["files"], "manifest.json", "VERSION"):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        assert (first / relative).read_text(encoding="utf-8") == (
            checked_in / relative
        ).read_text(encoding="utf-8")


def test_benchmark_rejects_an_underlying_failed_run() -> None:
    run = RunResult.model_construct(
        status="FAILED",
        summary=RunSummary.model_construct(incomplete=1),
        error=ErrorInfo(code="TARGET_PREPARE_ERROR", message="target could not be imported"),
    )

    with pytest.raises(
        BenchmarkExecutionError,
        match="output-only/campus: TARGET_PREPARE_ERROR: target could not be imported",
    ):
        _require_completed(run, profile="output-only", suite="campus")
