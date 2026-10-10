from pathlib import Path

import pytest

from agentqa.comparison import ComparisonError, compare_runs, load_run_artifact
from tests.fixtures.artifact_runs import create_run_artifact


def test_compare_aligns_cases_and_retains_removed_cases_in_pass_rate(
    tmp_path: Path,
) -> None:
    baseline_path = create_run_artifact(
        tmp_path / "baseline",
        "baseline-run",
        {"a": True, "b": False, "c": False, "d": True, "f": True},
        risks={"a": "high"},
        latency_ms=10,
        cost_usd=1,
    )
    candidate_path = create_run_artifact(
        tmp_path / "candidate",
        "candidate-run",
        {"a": False, "b": True, "c": False, "e": True, "f": True},
        risks={"a": "high"},
        latency_ms=15,
        cost_usd=1.5,
    )

    report = compare_runs(load_run_artifact(baseline_path), load_run_artifact(candidate_path))

    assert report.counts.model_dump() == {
        "new_regressions": 1,
        "fixed": 1,
        "persistent_failures": 1,
        "stable": 1,
        "new": 1,
        "removed": 1,
    }
    assert report.candidate.pass_rate == pytest.approx(3 / 5)
    assert report.candidate_effective_pass_rate == pytest.approx(3 / 6)
    assert report.candidate.high_risk_failures == 1
    assert report.deltas.p95_latency_ms.relative_increase == pytest.approx(0.5)
    assert report.deltas.average_cost_usd.relative_increase == pytest.approx(0.5)
    assert [case.test_case_id for case in report.cases] == sorted(
        case.test_case_id for case in report.cases
    )


def test_missing_metric_is_explicitly_unobservable(tmp_path: Path) -> None:
    baseline = create_run_artifact(tmp_path / "baseline", "baseline-run", {"a": True}, cost_usd=1)
    candidate = create_run_artifact(
        tmp_path / "candidate", "candidate-run", {"a": True}, cost_usd=None
    )

    report = compare_runs(load_run_artifact(baseline), load_run_artifact(candidate))

    assert report.deltas.average_cost_usd.status == "unobservable"
    assert report.deltas.average_cost_usd.relative_increase is None


def test_comparison_reports_mutation_score_and_replay_completeness(
    tmp_path: Path,
) -> None:
    baseline = create_run_artifact(
        tmp_path / "baseline",
        "baseline-run",
        {"mutation-case": True},
        mutation_status="SURVIVED",
        replay_hits=1,
        replay_misses=1,
    )
    candidate = create_run_artifact(
        tmp_path / "candidate",
        "candidate-run",
        {"mutation-case": False},
        mutation_status="KILLED",
        replay_hits=3,
        replay_misses=1,
    )

    report = compare_runs(load_run_artifact(baseline), load_run_artifact(candidate))

    assert report.baseline.mutation_score == 0.0
    assert report.candidate.mutation_score == 1.0
    assert report.deltas.mutation_score == 1.0
    assert report.baseline.replay.replay_hit_rate == 0.5
    assert report.candidate.replay.replay_hit_rate == 0.75
    assert report.deltas.replay_hit_rate == 0.25
    assert report.candidate.replay.replay_trials == 1


def test_loader_rejects_corrupted_checksummed_evidence(tmp_path: Path) -> None:
    run = create_run_artifact(tmp_path, "candidate-run", {"a": True})
    (run / "results.jsonl").write_text("corrupted\n", encoding="utf-8")

    with pytest.raises(ComparisonError, match="checksum mismatch"):
        load_run_artifact(run)
