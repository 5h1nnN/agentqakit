"""Run the T16 standalone golden path and controlled three-arm experiment."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import platform
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from agentqa.analysis.io import load_failure_analysis_input, write_analysis_report
from agentqa.application import (
    AnalysisError,
    FailureAnalysisService,
    GenerationError,
    GenerationService,
    RunExecutionConfig,
    execute_run,
)
from agentqa.comparison import compare_runs, load_run_artifact, write_report
from agentqa.evaluators import DeepSeekProvider, DeepSeekSettings, load_llm_environment
from agentqa.gates import evaluate_gate
from agentqa.generation import (
    load_generation_plan,
    write_generation_plan,
    write_generation_suite,
)
from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    CodeEvidence,
    GenerationGoal,
    GenerationReport,
    MutationSpec,
    QualityGateSpec,
    RunResult,
)
from agentqa.specs import hash_spec, load_spec_file, load_specs, normalize_spec
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "examples" / "showcase" / "campus_ops_agent"
AGENTQA = SHOWCASE / "agentqa"
VARIANTS = (
    "bug-retry",
    "bug-confirmation",
    "bug-authorization",
    "bug-injection",
    "bug-idempotency",
)


@dataclass(frozen=True, slots=True)
class Pricing:
    """USD per one million tokens, pinned in the experiment environment."""

    cache_hit: float
    cache_miss: float
    output: float
    source: str
    observed_at: str


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    manifest: Path
    with_code_evidence: bool


def _single(path: Path, model: type[Any]) -> Any:
    matches = [item.spec for item in load_spec_file(path) if isinstance(item.spec, model)]
    if len(matches) != 1:
        raise ValueError(f"{path} must contain exactly one {model.__name__}")
    return matches[0]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    path.with_name(f"{path.name}.sha256").write_text(f"{sha256_digest(payload)}\n")


def _checked_json(path: Path) -> dict[str, Any]:
    """Read one of this script's checksummed JSON checkpoints."""
    payload = path.read_bytes()
    checksum_path = path.with_name(f"{path.name}.sha256")
    if not checksum_path.is_file() or checksum_path.read_text().strip() != sha256_digest(payload):
        raise ValueError(f"checkpoint checksum mismatch: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint must contain an object: {path}")
    return cast(dict[str, Any], value)


def _single_run(directory: Path) -> Any:
    runs = sorted((directory / "runs").glob("*/run.json"))
    if len(runs) != 1:
        raise ValueError(f"expected exactly one completed run under {directory}")
    return load_run_artifact(runs[0].parent)


def _yaml_documents(path: Path, specs: list[AgentTestSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump_all(
            [normalize_spec(spec) for spec in specs],
            allow_unicode=True,
            explicit_start=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )


def _tokens(report: GenerationReport) -> dict[str, int]:
    totals: dict[str, int] = {}
    for call in report.llm_calls:
        for key, value in call.token_usage.items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _cost(tokens: dict[str, int], pricing: Pricing) -> float:
    hit = tokens.get("prompt_cache_hit_tokens", 0)
    miss = tokens.get("prompt_cache_miss_tokens")
    prompt = tokens.get("prompt_tokens", 0)
    if miss is None:
        miss = max(prompt - hit, 0)
    completion = tokens.get("completion_tokens", 0)
    return round(
        (hit * pricing.cache_hit + miss * pricing.cache_miss + completion * pricing.output)
        / 1_000_000,
        8,
    )


def _tool_references(spec: AgentTestSpec) -> set[str]:
    tools = spec.expect.tools
    if tools is None:
        return set()
    names = set(tools.required) | set(tools.forbidden) | set(tools.argument_schemas)
    if isinstance(tools.min_calls, dict):
        names.update(tools.min_calls)
    if isinstance(tools.max_calls, dict):
        names.update(tools.max_calls)
    for order in tools.order:
        names.update((order.before, order.after))
    return names


async def _run(
    *,
    tests: Path,
    target: Path,
    manifest: Path,
    artifacts: Path,
    provider: DeepSeekProvider,
    mutations: Path | None = None,
    replay_mode: str = "off",
) -> RunResult:
    result = await execute_run(
        RunExecutionConfig(
            tests_path=tests,
            target_path=target,
            mutations_path=mutations,
            manifest_path=manifest,
            artifact_directory=artifacts,
            random_seed=1601,
            concurrency=1,
            trace_receiver=False,
            capture_trace_content=True,
            replay_mode=cast(Any, replay_mode),
            original_command="python scripts/run_t16.py",
            judge_provider=provider,
        )
    )
    if result.status != "COMPLETED" or result.summary.incomplete:
        raise RuntimeError(f"run {result.run_id} did not complete")
    return result


def _run_root(artifacts: Path, run: RunResult) -> Path:
    return artifacts / "runs" / run.run_id


def _defects(fixed: RunResult, variants: dict[str, RunResult]) -> dict[str, Any]:
    fixed_passed = {item.test_case_id for item in fixed.results if item.passed}
    details: dict[str, Any] = {}
    for variant, run in variants.items():
        failed = sorted(
            item.test_case_id
            for item in run.results
            if item.test_case_id in fixed_passed and not item.passed
        )
        details[variant] = {"detected": bool(failed), "detecting_cases": failed}
    detected = sum(bool(item["detected"]) for item in details.values())
    return {
        "detected": detected,
        "total": len(VARIANTS),
        "rate": detected / len(VARIANTS),
        "variants": details,
    }


def _coverage(report: GenerationReport, manifest: AgentManifest) -> dict[str, Any]:
    capability = [item for item in report.coverage if item.subject_type == "capability"]
    policy = [item for item in report.coverage if item.subject_type == "policy"]
    referenced = set().union(*(_tool_references(case.spec) for case in report.cases))
    risky = {item.name for item in manifest.tools if item.risk in {"medium", "high", "critical"}}

    def ratio(items: list[Any]) -> float | None:
        return None if not items else sum(bool(item.case_ids) for item in items) / len(items)

    return {
        "capability": ratio(capability),
        "policy": ratio(policy),
        "tool_risk": None if not risky else len(referenced & risky) / len(risky),
        "referenced_tools": sorted(referenced),
    }


def _matches_truth(candidate: Any, terms: list[str]) -> bool:
    text = " ".join((candidate.id, candidate.hypothesis, candidate.reasoning)).casefold()
    return any(term in text for term in terms)


def _approved_specs(
    report: GenerationReport, approve_high_risk: bool
) -> tuple[list[AgentTestSpec], list[dict[str, Any]]]:
    selected: list[AgentTestSpec] = []
    actions: list[dict[str, Any]] = []
    for case in report.cases:
        status = case.review_status
        decision = "execute"
        spec = case.spec
        if status == "REVIEW_REQUIRED" and not approve_high_risk:
            decision = "reject"
        elif status == "REVIEW_REQUIRED":
            labels = {**spec.metadata.labels, "review-status": "APPROVED"}
            spec = spec.model_copy(
                update={"metadata": spec.metadata.model_copy(update={"labels": labels})}
            )
            decision = "approve"
        if decision != "reject":
            selected.append(spec)
        actions.append(
            {
                "case_id": case.case_id,
                "risk": case.risk,
                "initial_status": status,
                "decision": decision,
                "manual_edits": 0,
            }
        )
    return selected, actions


async def _generate_arm(
    arm: Arm,
    *,
    output: Path,
    provider: DeepSeekProvider,
    pricing: Pricing,
    approve_high_risk: bool,
    max_generated_cases: int | None,
) -> dict[str, Any]:
    arm_root = output / "arms" / arm.name
    manifest = cast(AgentManifest, _single(arm.manifest, AgentManifest))
    goal = cast(
        GenerationGoal, _single(AGENTQA / "generation-goal-comparison.yaml", GenerationGoal)
    )
    if max_generated_cases is not None:
        goal = goal.model_copy(update={"max_cases": max_generated_cases})
    if arm.with_code_evidence:
        evidence_goal = cast(
            GenerationGoal, _single(AGENTQA / "generation-goal-code.yaml", GenerationGoal)
        )
        goal = goal.model_copy(update={"code_evidence": evidence_goal.code_evidence})
    mutations = [
        item.spec
        for item in load_specs(AGENTQA / "mutations.yaml")
        if isinstance(item.spec, MutationSpec)
    ]
    service = GenerationService(provider)
    generation_stage = "plan"
    plan = None
    try:
        plan_path = arm_root / "plan.json"
        if plan_path.is_file():
            plan = load_generation_plan(plan_path)
            if plan.goal.max_cases != goal.max_cases:
                raise GenerationError(
                    "persisted plan case budget does not match the requested experiment"
                )
        else:
            plan = await service.generate_plan(manifest, goal)
            write_generation_plan(plan, plan_path, workspace=output)
        generation_stage = "tests"
        report = await service.generate_tests(manifest, plan, mutations)
    except GenerationError as exc:
        usage: dict[str, int] = {}
        calls = [] if plan is None else [plan.llm_call]
        if exc.llm_call is not None:
            calls.append(exc.llm_call)
        for call in calls:
            for key, value in call.token_usage.items():
                usage[key] = usage.get(key, 0) + value
        failure = {
            "schema_version": "agentqa.t16-generation-failure/v1",
            "arm": arm.name,
            "stage": generation_stage,
            "code": exc.code,
            "message": exc.message,
            "llm_call": (None if exc.llm_call is None else exc.llm_call.model_dump(mode="json")),
        }
        _json(arm_root / "generation-failure.json", failure)
        return {
            "arm": arm.name,
            "manifest_hash": hash_spec(manifest),
            "context_hashes": [item.content_hash for item in goal.code_evidence],
            "generation_id": None,
            "generation_status": "FAILED_SAFE",
            "case_budget": goal.max_cases,
            "generation_error": {
                "stage": generation_stage,
                "code": exc.code,
                "message": exc.message,
            },
            "generated": 0,
            "valid": 0,
            "valid_rate": 0.0,
            "invalid_output_batches": (
                exc.llm_call.attempts
                if exc.code == "GENERATION_INVALID_OUTPUT" and exc.llm_call is not None
                else 0
            ),
            "provider_error_batches": (
                exc.llm_call.attempts
                if exc.code == "GENERATION_PROVIDER_ERROR" and exc.llm_call is not None
                else 0
            ),
            "duplicate_cases": 0,
            "hallucinated_tool_cases": 0,
            "ready": 0,
            "review_required": 0,
            "approved": 0,
            "rejected": 0,
            "manual_edits": 0,
            "executed_behavior_tests": 0,
            "generated_mutation_tests": 0,
            "coverage": {"capability": None, "policy": None, "tool_risk": None},
            "defect_detection": None,
            "mutation_score": None,
            "tokens": usage,
            "estimated_cost_usd": _cost(usage, pricing),
            "tokens_per_valid_test": None,
            "cost_per_valid_test_usd": None,
            "run_ids": {"fixed": None, "variants": {}, "mutations": None},
        }
    write_generation_suite(report, arm_root / "generated", workspace=output)
    approved, actions = _approved_specs(report, approve_high_risk)
    behavior = [spec for spec in approved if spec.mutation is None]
    mutation_cases = [spec for spec in approved if spec.mutation is not None]
    _yaml_documents(arm_root / "reviewed" / "behavior.yaml", behavior)
    if mutation_cases:
        _yaml_documents(arm_root / "reviewed" / "mutations.yaml", mutation_cases)
    with (arm_root / "review-actions.jsonl").open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action, ensure_ascii=False, sort_keys=True) + "\n")

    artifacts = arm_root / "artifacts"
    fixed: RunResult | None = None
    variants: dict[str, RunResult] = {}
    if behavior:
        fixed = await _run(
            tests=arm_root / "reviewed" / "behavior.yaml",
            target=AGENTQA / "targets" / "fixed.yaml",
            manifest=arm.manifest,
            artifacts=artifacts / "fixed",
            provider=provider,
        )
        for variant in VARIANTS:
            variants[variant] = await _run(
                tests=arm_root / "reviewed" / "behavior.yaml",
                target=AGENTQA / "targets" / f"{variant}.yaml",
                manifest=arm.manifest,
                artifacts=artifacts / variant,
                provider=provider,
            )
    mutation_run = None
    if mutation_cases:
        mutation_run = await _run(
            tests=arm_root / "reviewed" / "mutations.yaml",
            target=AGENTQA / "targets" / "fixed.yaml",
            manifest=arm.manifest,
            artifacts=artifacts / "mutations",
            provider=provider,
            mutations=AGENTQA / "mutations.yaml",
        )
    usage = _tokens(report)
    valid = len(report.cases)
    hallucinated = sum(
        bool(_tool_references(case.spec) - {item.name for item in manifest.tools})
        for case in report.cases
    )
    duplicate_names = valid - len({case.case_id for case in report.cases})
    return {
        "arm": arm.name,
        "manifest_hash": hash_spec(manifest),
        "context_hashes": report.code_evidence_hashes,
        "generation_id": report.generation_id,
        "generation_status": "COMPLETED",
        "case_budget": goal.max_cases,
        "generation_error": None,
        "generated": valid,
        "valid": valid,
        "valid_rate": 1.0,
        "invalid_output_batches": sum(call.attempts - 1 for call in report.llm_calls),
        "provider_error_batches": 0,
        "duplicate_cases": duplicate_names,
        "hallucinated_tool_cases": hallucinated,
        "ready": sum(case.review_status == "READY" for case in report.cases),
        "review_required": sum(case.review_status == "REVIEW_REQUIRED" for case in report.cases),
        "approved": sum(action["decision"] == "approve" for action in actions),
        "rejected": sum(action["decision"] == "reject" for action in actions),
        "manual_edits": 0,
        "executed_behavior_tests": len(behavior),
        "generated_mutation_tests": len(mutation_cases),
        "coverage": _coverage(report, manifest),
        "defect_detection": None if fixed is None else _defects(fixed, variants),
        "mutation_score": (
            None
            if mutation_run is None or mutation_run.mutation_score is None
            else mutation_run.mutation_score.overall.score
        ),
        "tokens": usage,
        "estimated_cost_usd": _cost(usage, pricing),
        "tokens_per_valid_test": usage.get("total_tokens", 0) / valid,
        "cost_per_valid_test_usd": _cost(usage, pricing) / valid,
        "run_ids": {
            "fixed": None if fixed is None else fixed.run_id,
            "variants": {name: run.run_id for name, run in variants.items()},
            "mutations": None if mutation_run is None else mutation_run.run_id,
        },
    }


async def _standalone(*, output: Path, provider: DeepSeekProvider) -> dict[str, Any]:
    root = output / "standalone"
    artifacts = root / "artifacts"
    manifest = AGENTQA / "manifest-v2.yaml"
    golden = await _run(
        tests=AGENTQA / "golden-tests",
        target=AGENTQA / "targets" / "fixed.yaml",
        manifest=manifest,
        artifacts=artifacts / "golden",
        provider=provider,
        replay_mode="record",
    )
    mutation = await _run(
        tests=AGENTQA / "mutation-tests",
        target=AGENTQA / "targets" / "fixed.yaml",
        manifest=manifest,
        artifacts=artifacts / "mutations",
        provider=provider,
        mutations=AGENTQA / "mutations.yaml",
        replay_mode="record",
    )
    portable_golden = [
        item.spec.model_copy(update={"target_ref": None})
        for item in load_specs(AGENTQA / "golden-tests")
        if isinstance(item.spec, AgentTestSpec)
    ]
    broken_tests = root / "inputs" / "portable-golden.yaml"
    _yaml_documents(broken_tests, portable_golden)
    broken = await _run(
        tests=broken_tests,
        target=AGENTQA / "targets" / "bug-authorization.yaml",
        manifest=manifest,
        artifacts=artifacts / "broken",
        provider=provider,
        replay_mode="record",
    )
    failed = next(item for item in broken.results if not item.passed)
    inputs = load_failure_analysis_input(_run_root(artifacts / "broken", broken), failed.trial_id)
    analysis = await FailureAnalysisService(provider).analyze(
        run_id=inputs.run_id,
        test_spec=inputs.test_spec,
        result=inputs.result,
        trace=inputs.trace,
        manifest=inputs.manifest,
        code_evidence=(),
    )
    analysis_path = write_analysis_report(
        analysis, root / "analysis" / f"{analysis.analysis_id}.json", workspace=output
    )
    baseline = load_run_artifact(_run_root(artifacts / "broken", broken))
    candidate = load_run_artifact(_run_root(artifacts / "golden", golden))
    comparison = compare_runs(baseline, candidate)
    comparison_path = write_report(root / "comparison.json", comparison)
    regression_gate = cast(
        QualityGateSpec, _single(AGENTQA / "regression-gate.yaml", QualityGateSpec)
    )
    mutation_gate = cast(QualityGateSpec, _single(AGENTQA / "mutation-gate.yaml", QualityGateSpec))
    regression_result = evaluate_gate(regression_gate, candidate, comparison)
    mutation_artifact = load_run_artifact(_run_root(artifacts / "mutations", mutation))
    mutation_result = evaluate_gate(mutation_gate, mutation_artifact)
    write_report(root / "regression-gate.json", regression_result)
    write_report(root / "mutation-gate.json", mutation_result)
    return {
        "golden_run_id": golden.run_id,
        "golden_pass_rate": golden.summary.pass_rate,
        "judge_calls": sum(
            evaluation.evaluator_type == "llm_judge"
            for result in golden.results
            for evaluation in result.evaluations
        ),
        "mutation_run_id": mutation.run_id,
        "mutation_score": None
        if mutation.mutation_score is None
        else mutation.mutation_score.overall.score,
        "broken_run_id": broken.run_id,
        "failure_trial_id": failed.trial_id,
        "failure_bundle": failed.failure_bundle_ref,
        "analysis_report": analysis_path.relative_to(output).as_posix(),
        "analysis_evidence_hash": analysis.evidence_hash,
        "comparison_report": comparison_path.relative_to(output).as_posix(),
        "regression_gate_passed": regression_result.passed,
        "mutation_gate_passed": mutation_result.passed,
    }


def _reconstruct_standalone(output: Path) -> dict[str, Any]:
    """Recover the completed standalone stage created before checkpoint support."""
    root = output / "standalone"
    artifacts = root / "artifacts"
    golden = _single_run(artifacts / "golden")
    mutation = _single_run(artifacts / "mutations")
    broken = _single_run(artifacts / "broken")
    failed = next(item for item in broken.results if not item.passed)
    analysis_paths = sorted((root / "analysis").glob("*.json"))
    if len(analysis_paths) != 1:
        raise ValueError("expected exactly one standalone analysis report")
    analysis = _checked_json(analysis_paths[0])
    regression_gate = _checked_json(root / "regression-gate.json")
    mutation_gate = _checked_json(root / "mutation-gate.json")
    return {
        "golden_run_id": golden.run_id,
        "golden_pass_rate": golden.summary.pass_rate,
        "judge_calls": sum(
            evaluation.evaluator_type == "llm_judge"
            for result in golden.results
            for evaluation in result.evaluations
        ),
        "mutation_run_id": mutation.run_id,
        "mutation_score": (
            None if mutation.mutation_score is None else mutation.mutation_score.overall.score
        ),
        "broken_run_id": broken.run_id,
        "failure_trial_id": failed.trial_id,
        "failure_bundle": failed.failure_bundle_ref,
        "analysis_report": analysis_paths[0].relative_to(output).as_posix(),
        "analysis_evidence_hash": analysis["evidence_hash"],
        "comparison_report": (root / "comparison.json").relative_to(output).as_posix(),
        "regression_gate_passed": bool(regression_gate["passed"]),
        "mutation_gate_passed": bool(mutation_gate["passed"]),
    }


async def _analysis_benchmark(
    *,
    output: Path,
    provider: DeepSeekProvider,
    pricing: Pricing,
    code_evidence: list[CodeEvidence],
) -> dict[str, Any]:
    root = output / "analysis-benchmark"
    specs = [
        item.spec.model_copy(
            update={
                "target_ref": None,
                "evaluators": [
                    evaluator for evaluator in item.spec.evaluators if evaluator.type != "llm_judge"
                ],
            }
        )
        for item in load_specs(AGENTQA / "golden-tests")
        if isinstance(item.spec, AgentTestSpec)
    ]
    tests = root / "inputs" / "portable-deterministic-golden.yaml"
    _yaml_documents(tests, specs)
    manifest_path = AGENTQA / "manifest-v2.yaml"
    registry = yaml.safe_load((AGENTQA / "defect-registry.yaml").read_text(encoding="utf-8"))
    defects = cast(list[dict[str, Any]], registry["defects"])
    rows: list[dict[str, Any]] = []
    total_usage: dict[str, int] = {}
    for defect in defects:
        variant = cast(str, defect["variant"])
        defect_id = cast(str, defect["id"])
        row_checkpoint = root / "rows" / f"{defect_id}.json"
        if row_checkpoint.is_file():
            row = _checked_json(row_checkpoint)
            rows.append(row)
            for key, value in cast(dict[str, int], row.get("tokens", {})).items():
                total_usage[key] = total_usage.get(key, 0) + value
            continue
        artifact_root = root / "artifacts" / variant
        if (artifact_root / "runs").is_dir() and any((artifact_root / "runs").iterdir()):
            run_artifact = _single_run(artifact_root)
            run_id = run_artifact.run_id
            run_results = run_artifact.results
        else:
            run = await _run(
                tests=tests,
                target=AGENTQA / "targets" / f"{variant}.yaml",
                manifest=manifest_path,
                artifacts=artifact_root,
                provider=provider,
                replay_mode="record",
            )
            run_id = run.run_id
            run_results = run.results
        failed = next((item for item in run_results if not item.passed), None)
        if failed is None:
            row = {
                "defect_id": defect_id,
                "variant": variant,
                "detected": False,
                "analysis_status": "NOT_APPLICABLE",
                "top1": False,
                "top3": False,
                "analysis_report": None,
                "tokens": {},
            }
            _json(row_checkpoint, row)
            rows.append(row)
            continue
        inputs = load_failure_analysis_input(artifact_root / "runs" / run_id, failed.trial_id)
        try:
            report = await FailureAnalysisService(provider).analyze(
                run_id=inputs.run_id,
                test_spec=inputs.test_spec,
                result=inputs.result,
                trace=inputs.trace,
                manifest=inputs.manifest,
                code_evidence=code_evidence,
            )
        except AnalysisError as exc:
            usage = {} if exc.llm_call is None else dict(exc.llm_call.token_usage)
            row = {
                "defect_id": defect_id,
                "variant": variant,
                "detected": True,
                "analysis_status": "FAILED_SAFE",
                "analysis_error": {"code": exc.code, "message": exc.message},
                "failed_test": failed.test_case_id,
                "trial_id": failed.trial_id,
                "top1": False,
                "top3": False,
                "analysis_report": None,
                "tokens": usage,
            }
            _json(row_checkpoint, row)
            rows.append(row)
            for key, value in usage.items():
                total_usage[key] = total_usage.get(key, 0) + value
            continue
        report_path = write_analysis_report(
            report,
            root / "reports" / f"{defect_id}.json",
            workspace=output,
        )
        for key, value in report.llm_call.token_usage.items():
            total_usage[key] = total_usage.get(key, 0) + value
        terms = [str(term).casefold() for term in defect["analysisTruth"].get("matchTerms", [])]

        row = {
            "defect_id": defect_id,
            "variant": variant,
            "detected": True,
            "analysis_status": "COMPLETED",
            "failed_test": failed.test_case_id,
            "trial_id": failed.trial_id,
            "top1": _matches_truth(report.root_causes[0], terms),
            "top3": any(_matches_truth(item, terms) for item in report.root_causes[:3]),
            "candidate_ids": [item.id for item in report.root_causes[:3]],
            "match_terms": terms,
            "analysis_report": report_path.relative_to(output).as_posix(),
            "evidence_hash": report.evidence_hash,
            "tokens": dict(report.llm_call.token_usage),
        }
        _json(row_checkpoint, row)
        rows.append(row)
    analyzed = [row for row in rows if row.get("analysis_status") == "COMPLETED"]
    result = {
        "method": (
            "case-insensitive match against held-out defect-registry analysisTruth.matchTerms"
        ),
        "total_defects": len(rows),
        "analyzed": len(analyzed),
        "analysis_failures": sum(row.get("analysis_status") == "FAILED_SAFE" for row in rows),
        "top1_hits": sum(bool(row["top1"]) for row in analyzed),
        "top3_hits": sum(bool(row["top3"]) for row in analyzed),
        "top1_accuracy": (
            None if not analyzed else sum(bool(row["top1"]) for row in analyzed) / len(analyzed)
        ),
        "top3_accuracy": (
            None if not analyzed else sum(bool(row["top3"]) for row in analyzed) / len(analyzed)
        ),
        "tokens": total_usage,
        "estimated_cost_usd": _cost(total_usage, pricing),
        "defects": rows,
    }
    _json(root / "scores.json", result)
    return result


async def run(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    has_output = output.exists() and any(output.iterdir())
    if has_output and not args.resume:
        raise ValueError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    load_llm_environment(Path(args.env_file))
    settings = DeepSeekSettings.from_environment()
    if args.llm_max_tokens is not None:
        settings = replace(settings, max_tokens=args.llm_max_tokens)
    if args.llm_timeout_seconds is not None:
        settings = replace(settings, timeout_seconds=args.llm_timeout_seconds)
    provider = DeepSeekProvider(settings)
    pricing = Pricing(
        cache_hit=args.cache_hit_price,
        cache_miss=args.cache_miss_price,
        output=args.output_price,
        source=args.pricing_source,
        observed_at=args.pricing_observed_at,
    )
    if has_output:
        environment = _checked_json(output / "environment.json")
        provider_changed = environment.get("provider") != provider.name
        model_changed = environment.get("model") != provider.model
        if provider_changed or model_changed:
            raise ValueError("resume provider/model does not match the recorded environment")
        if environment.get("approve_high_risk") != args.approve_high_risk:
            raise ValueError("resume approval mode does not match the recorded environment")
    else:
        environment = {
            "schema_version": "agentqa.t16-environment/v1",
            "created_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "provider": provider.name,
            "model": provider.model,
            "parameters": dict(provider.parameters),
            "random_seed": 1601,
            "approve_high_risk": args.approve_high_risk,
            "pricing_usd_per_million_tokens": asdict(pricing),
            "showcase_source_hash": sha256_digest(
                b"".join(path.read_bytes() for path in sorted((SHOWCASE / "src").rglob("*.py")))
            ),
        }
        _json(output / "environment.json", environment)
    standalone_checkpoint = output / "standalone" / "stage-summary.json"
    if standalone_checkpoint.is_file():
        standalone = _checked_json(standalone_checkpoint)
    elif has_output and (output / "standalone" / "artifacts").is_dir():
        standalone = _reconstruct_standalone(output)
        _json(standalone_checkpoint, standalone)
    else:
        standalone = await _standalone(output=output, provider=provider)
        _json(standalone_checkpoint, standalone)
    arms = (
        Arm("minimal-manifest", AGENTQA / "manifest-minimal.yaml", False),
        Arm("manifest-v2", AGENTQA / "manifest-v2.yaml", False),
        Arm("manifest-v2-code-evidence", AGENTQA / "manifest-v2.yaml", True),
    )
    metrics: list[dict[str, Any]] = []
    for arm in arms:
        checkpoint = output / "arms" / arm.name / "stage-summary.json"
        if checkpoint.is_file():
            metric = _checked_json(checkpoint)
        else:
            metric = await _generate_arm(
                arm,
                output=output,
                provider=provider,
                pricing=pricing,
                approve_high_risk=args.approve_high_risk,
                max_generated_cases=args.max_generated_cases,
            )
            _json(checkpoint, metric)
        metric.setdefault("provider_error_batches", 0)
        metrics.append(metric)
    evidence_goal = cast(
        GenerationGoal, _single(AGENTQA / "generation-goal-code.yaml", GenerationGoal)
    )
    analysis_checkpoint = output / "analysis-benchmark" / "stage-summary.json"
    if analysis_checkpoint.is_file():
        analysis_benchmark = _checked_json(analysis_checkpoint)
    else:
        analysis_benchmark = await _analysis_benchmark(
            output=output,
            provider=provider,
            pricing=pricing,
            code_evidence=evidence_goal.code_evidence,
        )
        _json(analysis_checkpoint, analysis_benchmark)
    summary = {
        "schema_version": "agentqa.t16-summary/v1",
        "generation_provider_parameters": dict(provider.parameters),
        "standalone": standalone,
        "arms": metrics,
        "analysis_benchmark": analysis_benchmark,
    }
    _json(output / "summary.json", summary)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "arm",
            "valid",
            "valid_rate",
            "invalid_output_batches",
            "provider_error_batches",
            "duplicate_cases",
            "hallucinated_tool_cases",
            "ready",
            "review_required",
            "approved",
            "rejected",
            "executed_behavior_tests",
            "generated_mutation_tests",
            "mutation_score",
            "tokens_per_valid_test",
            "cost_per_valid_test_usd",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow({key: metric.get(key) for key in fields})
    shutil.copy2(AGENTQA / "defect-registry.yaml", output / "defect-registry.yaml")
    print(f"T16 completed: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmark/results/t16-deepseek-20261014")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--approve-high-risk", action="store_true")
    parser.add_argument(
        "--max-generated-cases",
        type=int,
        choices=range(1, 51),
        metavar="N",
        help="override the comparison goal case budget (1-50)",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        choices=range(64, 16_385),
        metavar="N",
        help="override the provider output budget (64-16384)",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        metavar="SECONDS",
        help="override the provider timeout (1-600 seconds)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify and reuse completed stage checkpoints in a non-empty output directory",
    )
    parser.add_argument("--cache-hit-price", type=float, default=0.003)
    parser.add_argument("--cache-miss-price", type=float, default=0.15)
    parser.add_argument("--output-price", type=float, default=0.60)
    parser.add_argument(
        "--pricing-source", default="https://api-docs.deepseek.com/quick_start/pricing/"
    )
    parser.add_argument("--pricing-observed-at", default="2026-10-14")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
