from pathlib import Path

import pytest

from agentqa.comparison import compare_runs, load_run_artifact
from agentqa.gates import GateError, evaluate_gate
from agentqa.models import QualityGateSpec
from tests.fixtures.artifact_runs import create_run_artifact


def _gate(rules: dict[str, object]) -> QualityGateSpec:
    return QualityGateSpec.model_validate(
        {
            "apiVersion": "agentqa.dev/v1alpha1",
            "kind": "QualityGate",
            "metadata": {"name": "ci"},
            "rules": rules,
        }
    )


def test_gate_evaluates_absolute_and_relative_rules(tmp_path: Path) -> None:
    baseline_path = create_run_artifact(
        tmp_path / "baseline",
        "baseline-run",
        {"a": True, "removed": True},
        latency_ms=10,
        cost_usd=1,
    )
    candidate_path = create_run_artifact(
        tmp_path / "candidate",
        "candidate-run",
        {"a": True, "new": True},
        latency_ms=12,
        cost_usd=1.2,
    )
    baseline = load_run_artifact(baseline_path)
    candidate = load_run_artifact(candidate_path)
    comparison = compare_runs(baseline, candidate)
    spec = _gate(
        {
            "minPassRate": 0.6,
            "maxNewRegressions": 0,
            "maxHighRiskFailures": 0,
            "maxP95LatencyIncrease": 0.25,
            "maxAverageCostIncrease": 0.25,
            "requireReplayBundleForFailures": True,
        }
    )

    result = evaluate_gate(spec, candidate, comparison)

    assert result.passed
    assert result.exit_code == 0
    assert all(rule.passed for rule in result.rules)


def test_removed_case_cannot_inflate_gate_pass_rate(tmp_path: Path) -> None:
    baseline = load_run_artifact(
        create_run_artifact(tmp_path / "baseline", "baseline-run", {"kept": True, "removed": True})
    )
    candidate = load_run_artifact(
        create_run_artifact(tmp_path / "candidate", "candidate-run", {"kept": True})
    )

    result = evaluate_gate(
        _gate({"minPassRate": 0.75}),
        candidate,
        compare_runs(baseline, candidate),
    )

    assert not result.passed
    assert result.rules[0].actual == 0.5


def test_configured_unobservable_metric_fails(tmp_path: Path) -> None:
    baseline = load_run_artifact(
        create_run_artifact(tmp_path / "baseline", "baseline-run", {"a": True}, cost_usd=1)
    )
    candidate = load_run_artifact(
        create_run_artifact(tmp_path / "candidate", "candidate-run", {"a": True}, cost_usd=None)
    )

    result = evaluate_gate(
        _gate({"maxAverageCostIncrease": 0.1}),
        candidate,
        compare_runs(baseline, candidate),
    )

    assert not result.passed
    assert result.rules[0].actual == {
        "status": "unobservable",
        "baseline": 1.0,
        "candidate": None,
    }


def test_missing_mutation_score_fails_and_missing_baseline_is_error(
    tmp_path: Path,
) -> None:
    candidate = load_run_artifact(create_run_artifact(tmp_path, "candidate-run", {"a": True}))

    result = evaluate_gate(_gate({"minMutationScore": 0.8}), candidate)
    assert not result.passed
    assert result.rules[0].actual is None

    with pytest.raises(GateError, match="require --baseline"):
        evaluate_gate(_gate({"maxNewRegressions": 0}), candidate)


def test_failed_trial_without_verified_replay_bundle_fails_gate(
    tmp_path: Path,
) -> None:
    candidate = load_run_artifact(
        create_run_artifact(tmp_path, "candidate-run", {"high-risk-case": False})
    )

    result = evaluate_gate(
        _gate({"requireReplayBundleForFailures": True}),
        candidate,
    )

    assert not result.passed
    assert result.rules[0].actual == 0.0
    assert "coverage" in (result.rules[0].message or "")
