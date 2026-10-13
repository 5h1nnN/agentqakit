"""Verified loading and metric extraction for completed run artifacts."""

import json
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from agentqa.errors import AgentQAError
from agentqa.models import (
    AgentTestSpec,
    MutationScoreReport,
    ReplayCompleteness,
    RunMetrics,
    RunSummary,
    TestResult,
    TraceRecord,
)
from agentqa.replay.bundle import extract_failure_bundle
from agentqa.replay.store import CassetteFormatError
from agentqa.runner.aggregation import aggregate_run, summarize_metric
from agentqa.specs.hashing import canonical_json_bytes
from agentqa.storage import ArtifactError, ArtifactWriter
from agentqa.traceview import TraceView


class ComparisonError(AgentQAError):
    """Raised when run evidence cannot be safely compared."""

    def __init__(self, message: str, *, code: str = "COMPARISON_ERROR") -> None:
        super().__init__(message, code=code)


@dataclass(frozen=True, slots=True)
class RunArtifact:
    """Validated and internally consistent evidence from one completed run."""

    root: Path
    reader: ArtifactWriter
    run_id: str
    summary: RunSummary
    tests: tuple[AgentTestSpec, ...]
    results: tuple[TestResult, ...]
    mutation_score: MutationScoreReport | None


def _reader(run_directory: str | Path) -> ArtifactWriter:
    resolved = Path(run_directory).resolve()
    if resolved.parent.name != "runs":
        raise ComparisonError(
            "run directory must be an artifacts/runs/<run-id> directory",
            code="COMPARISON_PATH_ERROR",
        )
    reader = ArtifactWriter(resolved.parent.parent, resolved.name)
    if reader.root != resolved:
        raise ComparisonError(
            "run directory could not be resolved safely",
            code="COMPARISON_PATH_ERROR",
        )
    return reader


def _document(reader: ArtifactWriter, name: str) -> dict[str, Any]:
    if not reader.verify_checksum(name):
        raise ComparisonError(
            f"{name} checksum mismatch",
            code="COMPARISON_CHECKSUM_ERROR",
        )
    try:
        value = json.loads((reader.root / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ComparisonError(
            f"{name} is not valid JSON",
            code="COMPARISON_FORMAT_ERROR",
        ) from None
    if not isinstance(value, dict):
        raise ComparisonError(
            f"{name} must contain a JSON object",
            code="COMPARISON_FORMAT_ERROR",
        )
    document = cast(dict[str, Any], value)
    document.pop("schema_version", None)
    return document


def load_run_artifact(run_directory: str | Path) -> RunArtifact:
    """Load a checksummed, completed run and reject inconsistent derived data."""
    reader = _reader(run_directory)
    try:
        run_document = _document(reader, "run.json")
        summary_document = _document(reader, "summary.json")
        test_document = _document(reader, "resolved-tests.json")
        try:
            result_records = reader.read_result_records()
        except ArtifactError as exc:
            code = (
                "COMPARISON_CHECKSUM_ERROR"
                if exc.code == "ARTIFACT_CHECKSUM_ERROR"
                else "COMPARISON_FORMAT_ERROR"
            )
            raise ComparisonError(exc.message, code=code) from None
        tests = tuple(AgentTestSpec.model_validate(value) for value in test_document["tests"])
        results = tuple(
            TestResult.model_validate_json(json.dumps(value)) for value in result_records
        )
        summary = RunSummary.model_validate(summary_document["summary"])
        mutation_value = summary_document.get("mutation_score")
        mutation_score = (
            None if mutation_value is None else MutationScoreReport.model_validate(mutation_value)
        )
    except (KeyError, TypeError, ValidationError, ValueError):
        raise ComparisonError(
            "run artifacts do not match the expected schema",
            code="COMPARISON_FORMAT_ERROR",
        ) from None

    run_id = summary_document.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ComparisonError("summary.json has no run_id", code="COMPARISON_FORMAT_ERROR")
    if run_id != reader.root.name or run_document.get("run_id") != run_id:
        raise ComparisonError(
            "run identifiers do not match the artifact directory",
            code="COMPARISON_FORMAT_ERROR",
        )
    if summary_document.get("status") != "COMPLETED" or run_document.get("status") != ("COMPLETED"):
        raise ComparisonError(
            "only completed runs can be compared",
            code="COMPARISON_INCOMPLETE_RUN",
        )
    if summary.incomplete or summary_document.get("exit_code") not in {0, 1}:
        raise ComparisonError(
            "only complete run summaries can be compared",
            code="COMPARISON_INCOMPLETE_RUN",
        )

    test_ids = [test.metadata.name for test in tests]
    result_ids = [result.trial_id for result in results]
    if len(test_ids) != len(set(test_ids)) or len(result_ids) != len(set(result_ids)):
        raise ComparisonError(
            "test case and trial identifiers must be unique",
            code="COMPARISON_FORMAT_ERROR",
        )
    if {case.test_case_id for case in summary.cases} != set(test_ids):
        raise ComparisonError(
            "summary cases do not match resolved tests",
            code="COMPARISON_FORMAT_ERROR",
        )
    if any(result.test_case_id not in set(test_ids) for result in results):
        raise ComparisonError(
            "results contain an unknown test case",
            code="COMPARISON_FORMAT_ERROR",
        )
    if aggregate_run(tests, results) != summary:
        raise ComparisonError(
            "summary cannot be recomputed from resolved tests and results",
            code="COMPARISON_FORMAT_ERROR",
        )

    score_path = reader.root / "mutation-score.json"
    if mutation_score is not None:
        try:
            persisted_score = MutationScoreReport.model_validate(
                _document(reader, "mutation-score.json")
            )
        except ValidationError:
            raise ComparisonError(
                "mutation-score.json does not match the expected schema",
                code="COMPARISON_FORMAT_ERROR",
            ) from None
        if persisted_score != mutation_score:
            raise ComparisonError(
                "mutation score evidence is inconsistent",
                code="COMPARISON_FORMAT_ERROR",
            )
    elif score_path.exists():
        raise ComparisonError(
            "unexpected mutation-score.json without a summary score",
            code="COMPARISON_FORMAT_ERROR",
        )

    return RunArtifact(
        root=reader.root,
        reader=reader,
        run_id=run_id,
        summary=summary,
        tests=tests,
        results=results,
        mutation_score=mutation_score,
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return None
    return float(value)


def _observed_metric(result: TestResult, key: str) -> float | None:
    direct = _number(result.target_result.metadata.get(key))
    if direct is not None:
        return direct
    for evaluation in result.evaluations:
        if evaluation.evaluator_type == "budget":
            observed = _number(evaluation.metadata.get(key))
            if observed is not None:
                return observed
    return None


def _trace_for(artifact: RunArtifact, result: TestResult) -> TraceRecord | None:
    reference = result.trace_ref
    if reference is None:
        return None
    relative = Path(reference)
    if relative.parent != Path("traces") or relative.name != f"{result.trace_id}.json":
        return None
    if not artifact.reader.verify_checksum(relative):
        return None
    try:
        document = json.loads((artifact.root / relative).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            return None
        document.pop("schema_version", None)
        return TraceRecord.model_validate(document)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError):
        return None


def _trace_behavior_metrics(trace: TraceRecord) -> dict[str, float]:
    view = TraceView(trace)
    signatures: list[bytes] = []
    seen_tools: set[bytes] = set()
    duplicate_tools = 0
    for call in view.tool_calls:
        signature = canonical_json_bytes({"name": call.name, "arguments": call.arguments})
        if signature in seen_tools:
            duplicate_tools += 1
        seen_tools.add(signature)
    for event in view.internal_events:
        if event.kind not in {"LLM", "TOOL"}:
            continue
        signatures.append(
            canonical_json_bytes(
                {
                    "kind": event.kind,
                    "name": event.name,
                    "input": event.input,
                    "output": event.output,
                    "state_diff": event.state_diff,
                }
            )
        )
    no_progress = sum(current == previous for previous, current in pairwise(signatures))
    retries = 0.0
    retry_observed = False
    for event in view.internal_events:
        for key in ("retry_count", "retries", "agentqa.retry.count"):
            value = _number(event.attributes.get(key))
            if value is not None:
                retries += value
                retry_observed = True
                break
    return {
        "llm_calls": float(len(view.llm_calls)),
        "tool_calls": float(len(view.tool_calls)),
        "duplicate_tool_calls": float(duplicate_tools),
        "no_progress_loops": float(no_progress),
        **({"retries": retries} if retry_observed else {}),
    }


def _behavior_metric(artifact: RunArtifact, result: TestResult, key: str) -> float | None:
    observed = _observed_metric(result, key)
    if observed is not None:
        return observed
    trace = _trace_for(artifact, result)
    if trace is None or trace.status != "complete":
        return None
    return _trace_behavior_metrics(trace).get(key)


def _complete_metric(results: tuple[TestResult, ...], key: str) -> list[float]:
    values = [_observed_metric(result, key) for result in results]
    if any(value is None for value in values):
        return []
    return cast(list[float], values)


def _complete_behavior_metric(
    artifact: RunArtifact,
    key: str,
) -> list[float]:
    values = [_behavior_metric(artifact, result, key) for result in artifact.results]
    if any(value is None for value in values):
        return []
    return cast(list[float], values)


def _has_valid_failure_bundle(artifact: RunArtifact, result: TestResult) -> bool:
    reference = result.failure_bundle_ref
    if reference is None:
        return False
    relative = Path(reference)
    if relative.is_absolute() or relative.parent != Path("failures") or relative.suffix != ".zip":
        return False
    try:
        if not artifact.reader.verify_checksum(relative):
            return False
        with tempfile.TemporaryDirectory(prefix="agentqa-bundle-check-") as directory:
            extract_failure_bundle(artifact.root / relative, Path(directory))
        return True
    except (ArtifactError, CassetteFormatError, OSError):
        return False


def metrics_for(artifact: RunArtifact) -> RunMetrics:
    """Derive comparable metrics without inventing missing observations."""
    latencies = [
        (result.target_result.ended_at - result.target_result.started_at).total_seconds() * 1_000
        for result in artifact.results
    ]
    failed_results = [result for result in artifact.results if not result.passed]
    valid_bundles = sum(_has_valid_failure_bundle(artifact, result) for result in failed_results)
    replay_stats = [result.replay for result in artifact.results if result.replay is not None]
    replay_hits = sum(stats.hits for stats in replay_stats)
    replay_misses = sum(stats.misses for stats in replay_stats)
    replay_attempts = replay_hits + replay_misses
    case_outcomes = {case.test_case_id: case.passed for case in artifact.summary.cases}
    high_risk_ids = {
        test.metadata.name
        for test in artifact.tests
        if test.metadata.labels.get("risk", "").strip().lower() in {"high", "critical"}
    }
    mutation_score = (
        None if artifact.mutation_score is None else artifact.mutation_score.overall.score
    )
    return RunMetrics(
        total=artifact.summary.total,
        passed=artifact.summary.passed,
        failed=artifact.summary.failed,
        pass_rate=artifact.summary.pass_rate,
        mutation_score=mutation_score,
        high_risk_failures=sum(case_outcomes[test_id] is False for test_id in high_risk_ids),
        latency_ms=summarize_metric(latencies),
        total_tokens=summarize_metric(_complete_metric(artifact.results, "total_tokens")),
        cost_usd=summarize_metric(_complete_metric(artifact.results, "cost_usd")),
        llm_calls=summarize_metric(_complete_behavior_metric(artifact, "llm_calls")),
        tool_calls=summarize_metric(_complete_behavior_metric(artifact, "tool_calls")),
        retries=summarize_metric(_complete_behavior_metric(artifact, "retries")),
        duplicate_tool_calls=summarize_metric(
            _complete_behavior_metric(artifact, "duplicate_tool_calls")
        ),
        no_progress_loops=summarize_metric(
            _complete_behavior_metric(artifact, "no_progress_loops")
        ),
        replay=ReplayCompleteness(
            total_trials=len(artifact.results),
            replay_trials=sum(stats.mode == "replay" for stats in replay_stats),
            replay_hits=replay_hits,
            replay_misses=replay_misses,
            live_fallbacks=sum(stats.live_fallbacks for stats in replay_stats),
            replay_hit_rate=(None if replay_attempts == 0 else replay_hits / replay_attempts),
            failed_trials=len(failed_results),
            failure_bundles=valid_bundles,
            failure_bundle_coverage=(
                1.0 if not failed_results else valid_bundles / len(failed_results)
            ),
        ),
    )
