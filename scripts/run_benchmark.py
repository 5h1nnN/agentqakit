"""Run AgentQA-Bench baselines and ablations from one reproducible command."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

# This script is intentionally runnable by absolute path from outside the
# repository. In that mode Python adds ``scripts/`` rather than the repository
# root to sys.path, so repository-local benchmark targets under ``examples``
# would otherwise be unavailable even when AgentQAKit itself is installed.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import yaml  # noqa: E402

from agentqa.application import RunExecutionConfig, execute_run  # noqa: E402
from agentqa.evaluators import JudgeProvider  # noqa: E402
from agentqa.evaluators.deepseek import (  # noqa: E402
    DeepSeekJudgeProvider,
    load_judge_environment,
)
from agentqa.models import RunResult, TestResult  # noqa: E402


@dataclass(frozen=True, slots=True)
class Suite:
    name: str
    cases: Path
    target: Path


@dataclass(frozen=True, slots=True)
class Configuration:
    name: str
    evaluator_types: frozenset[str]
    replay_mode: Literal["off", "record", "replay"]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    source: Path
    root: Path
    benchmark_version: str
    random_seed: int
    capture_trace_content: bool
    judge_env_file: Path
    suites: tuple[Suite, ...]
    mutations: Path
    configurations: tuple[Configuration, ...]


class BenchmarkExecutionError(RuntimeError):
    """Raised when an underlying AgentQAKit run cannot produce evidence."""


def _require_completed(run: RunResult, *, profile: str, suite: str) -> None:
    if run.status == "COMPLETED" and run.summary.incomplete == 0:
        return
    detail = "run did not complete"
    if run.error is not None:
        detail = f"{run.error.code}: {run.error.message}"
    raise BenchmarkExecutionError(f"{profile}/{suite}: {detail}")


def _workspace_path(root: Path, raw: Any, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"{field} must stay inside the repository") from None
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def load_config(path: Path) -> BenchmarkConfig:
    source = path.resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark config must be a mapping")
    root = source.parent.parent.resolve()
    suites_raw = raw.get("suites")
    if not isinstance(suites_raw, list) or not suites_raw:
        raise ValueError("benchmark config requires suites")
    suites: list[Suite] = []
    for item in suites_raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("each benchmark suite requires a name")
        suites.append(
            Suite(
                name=item["name"],
                cases=_workspace_path(root, item.get("cases"), field="suite.cases"),
                target=_workspace_path(root, item.get("target"), field="suite.target"),
            )
        )
    configurations_raw = raw.get("configurations")
    if not isinstance(configurations_raw, list) or not configurations_raw:
        raise ValueError("benchmark config requires configurations")
    configurations: list[Configuration] = []
    for item in configurations_raw:
        if not isinstance(item, dict):
            raise ValueError("each configuration must be a mapping")
        name = item.get("name")
        evaluator_types = item.get("evaluatorTypes")
        replay_mode = item.get("replayMode", "off")
        if not isinstance(name, str) or not name:
            raise ValueError("each configuration requires a name")
        if not isinstance(evaluator_types, list) or not all(
            isinstance(value, str) and value for value in evaluator_types
        ):
            raise ValueError(f"configuration {name} requires evaluatorTypes")
        if replay_mode not in {"off", "record", "replay"}:
            raise ValueError(f"configuration {name} has invalid replayMode")
        configurations.append(
            Configuration(name, frozenset(evaluator_types), cast(Any, replay_mode))
        )
    judge_env_raw = raw.get("judgeEnvFile", ".env.local")
    if not isinstance(judge_env_raw, str) or not judge_env_raw:
        raise ValueError("judgeEnvFile must be a non-empty path")
    return BenchmarkConfig(
        source=source,
        root=root,
        benchmark_version=str(raw.get("benchmarkVersion") or "unknown"),
        random_seed=int(raw.get("randomSeed", 0)),
        capture_trace_content=bool(raw.get("captureTraceContent", False)),
        judge_env_file=(root / judge_env_raw).resolve(),
        suites=tuple(suites),
        mutations=_workspace_path(root, raw.get("mutations"), field="mutations"),
        configurations=tuple(configurations),
    )


def _read_documents(path: Path) -> list[dict[str, Any]]:
    documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    if not all(isinstance(document, dict) for document in documents):
        raise ValueError(f"benchmark cases must be mappings: {path}")
    return cast(list[dict[str, Any]], documents)


def _profile_documents(
    documents: list[dict[str, Any]],
    evaluator_types: frozenset[str],
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for document in documents[:limit]:
        copied = json.loads(json.dumps(document, ensure_ascii=False))
        evaluators = copied.get("evaluators", [])
        copied["evaluators"] = [
            evaluator
            for evaluator in evaluators
            if isinstance(evaluator, dict) and evaluator.get("type") in evaluator_types
        ]
        if not copied["evaluators"]:
            copied["evaluators"] = [{"type": "schema"}]
        selected.append(copied)
    return selected


def _write_documents(path: Path, documents: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(
        path,
        yaml.safe_dump_all(
            documents,
            allow_unicode=True,
            explicit_start=True,
            sort_keys=False,
            width=100,
        ),
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _case_metadata(documents: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        str(document["metadata"]["name"]): cast(dict[str, str], document["metadata"]["labels"])
        for document in documents
    }


async def _execute(
    *,
    config: BenchmarkConfig,
    suite: Suite,
    profile: Configuration,
    cases_path: Path,
    artifacts: Path,
    provider: JudgeProvider | None,
    replay_mode: Literal["off", "record", "replay"],
    replay_cassette: Path | None = None,
) -> RunResult:
    return await execute_run(
        RunExecutionConfig(
            tests_path=cases_path,
            target_path=suite.target,
            mutations_path=config.mutations,
            artifact_directory=artifacts,
            random_seed=config.random_seed,
            trace_receiver=False,
            trace_flush_timeout_seconds=0.05,
            capture_trace_content=config.capture_trace_content,
            replay_mode=replay_mode,
            replay_cassette=replay_cassette,
            generate_failure_bundles=True,
            original_command=f"scripts/run_benchmark.py:{profile.name}:{suite.name}",
            judge_provider=provider,
        )
    )


def _copy_failure(
    *, output: Path, artifacts: Path, run_id: str, profile: str, suite: str, result: TestResult
) -> str | None:
    if result.failure_bundle_ref is None:
        return None
    source = artifacts / "runs" / run_id / result.failure_bundle_ref
    if not source.is_file():
        return None
    destination = output / "failures" / profile / suite / f"{result.test_case_id}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    checksum = source.with_suffix(source.suffix + ".sha256")
    if checksum.is_file():
        shutil.copy2(checksum, destination.with_suffix(destination.suffix + ".sha256"))
    return str(destination.relative_to(output))


def _raw_row(
    *,
    profile: str,
    suite: str,
    run: RunResult,
    result: TestResult,
    labels: dict[str, str],
    failure_ref: str | None,
) -> dict[str, Any]:
    return {
        "configuration": profile,
        "suite": suite,
        "runId": run.run_id,
        "testCaseId": result.test_case_id,
        "labels": labels,
        "passed": result.passed,
        "failureBundle": failure_ref,
        "result": result.model_dump(mode="json"),
    }


async def _standard_run(
    *,
    benchmark: BenchmarkConfig,
    suite: Suite,
    profile: Configuration,
    documents: list[dict[str, Any]],
    output: Path,
    provider: JudgeProvider | None,
) -> list[dict[str, Any]]:
    input_path = output / "inputs" / profile.name / f"{suite.name}.yaml"
    _write_documents(input_path, documents)
    artifacts = output / "artifacts" / profile.name / suite.name
    run = await _execute(
        config=benchmark,
        suite=suite,
        profile=profile,
        cases_path=input_path,
        artifacts=artifacts,
        provider=provider,
        replay_mode=profile.replay_mode,
    )
    _require_completed(run, profile=profile.name, suite=suite.name)
    labels = _case_metadata(documents)
    return [
        _raw_row(
            profile=profile.name,
            suite=suite.name,
            run=run,
            result=result,
            labels=labels[result.test_case_id],
            failure_ref=_copy_failure(
                output=output,
                artifacts=artifacts,
                run_id=run.run_id,
                profile=profile.name,
                suite=suite.name,
                result=result,
            ),
        )
        for result in run.results
    ]


async def _replay_run(
    *,
    benchmark: BenchmarkConfig,
    suite: Suite,
    profile: Configuration,
    documents: list[dict[str, Any]],
    output: Path,
    provider: JudgeProvider | None,
) -> list[dict[str, Any]]:
    record_input = output / "inputs" / profile.name / f"{suite.name}-record.yaml"
    _write_documents(record_input, documents)
    record_artifacts = output / "artifacts" / profile.name / f"{suite.name}-record"
    record = await _execute(
        config=benchmark,
        suite=suite,
        profile=profile,
        cases_path=record_input,
        artifacts=record_artifacts,
        provider=None,
        replay_mode="record",
    )
    _require_completed(record, profile=f"{profile.name}:record", suite=suite.name)
    cassettes = {
        result.test_case_id: (
            record_artifacts / "runs" / record.run_id / result.cassette_ref
            if result.cassette_ref is not None
            else None
        )
        for result in record.results
    }
    labels = _case_metadata(documents)
    rows: list[dict[str, Any]] = []
    for index, document in enumerate(documents, start=1):
        test_case_id = str(document["metadata"]["name"])
        cassette = cassettes.get(test_case_id)
        if cassette is None or not cassette.is_file():
            raise RuntimeError(f"record phase did not produce cassette for {test_case_id}")
        case_path = output / "inputs" / profile.name / suite.name / f"{index:03d}.yaml"
        _write_documents(case_path, [document])
        artifacts = output / "artifacts" / profile.name / suite.name / f"{index:03d}"
        run = await _execute(
            config=benchmark,
            suite=suite,
            profile=profile,
            cases_path=case_path,
            artifacts=artifacts,
            provider=provider,
            replay_mode="replay",
            replay_cassette=cassette,
        )
        _require_completed(run, profile=profile.name, suite=suite.name)
        result = run.results[0]
        rows.append(
            _raw_row(
                profile=profile.name,
                suite=suite.name,
                run=run,
                result=result,
                labels=labels[test_case_id],
                failure_ref=_copy_failure(
                    output=output,
                    artifacts=artifacts,
                    run_id=run.run_id,
                    profile=profile.name,
                    suite=suite.name,
                    result=result,
                ),
            )
        )
    return rows


def _metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    names = sorted({str(row["configuration"]) for row in rows})
    for name in names:
        selected = [row for row in rows if row["configuration"] == name]
        evaluations = [
            evaluation for row in selected for evaluation in row["result"].get("evaluations", [])
        ]
        mutations = [
            application
            for row in selected
            for application in row["result"].get("mutation_applications", [])
        ]
        total = len(selected)
        passed = sum(bool(row["passed"]) for row in selected)
        failed_mutation_cases = sum(
            bool(row["result"].get("mutation_applications")) and not bool(row["passed"])
            for row in selected
        )
        mutation_cases = sum(bool(row["result"].get("mutation_applications")) for row in selected)
        metrics.append(
            {
                "configuration": name,
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": 0.0 if total == 0 else passed / total,
                "judge_pass": sum(
                    item.get("evaluator_type") == "llm_judge" and item.get("status") == "PASS"
                    for item in evaluations
                ),
                "judge_error": sum(
                    item.get("evaluator_type") == "llm_judge" and item.get("status") == "ERROR"
                    for item in evaluations
                ),
                "judge_skipped": sum(
                    item.get("evaluator_type") == "llm_judge" and item.get("status") == "SKIPPED"
                    for item in evaluations
                ),
                "mutation_applications": len(mutations),
                "mutation_detection_rate": (
                    0.0 if mutation_cases == 0 else failed_mutation_cases / mutation_cases
                ),
                "replay_hits": sum(
                    int(row["result"].get("replay", {}).get("hits", 0)) for row in selected
                ),
                "replay_misses": sum(
                    int(row["result"].get("replay", {}).get("misses", 0)) for row in selected
                ),
            }
        )
    return metrics


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _mutation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for row in rows:
        applications = row["result"].get("mutation_applications", [])
        for application in applications:
            matrix.append(
                {
                    "configuration": row["configuration"],
                    "suite": row["suite"],
                    "test_case_id": row["testCaseId"],
                    "mutation_id": application.get("mutation_id"),
                    "operation": application.get("operation"),
                    "triggered": application.get("triggered"),
                    "detected": not bool(row["passed"]),
                }
            )
    return matrix


def _write_chart(path: Path, metrics: list[dict[str, Any]]) -> None:
    width = 960
    row_height = 52
    height = 80 + row_height * len(metrics)
    bars: list[str] = []
    for index, metric in enumerate(metrics):
        y = 46 + index * row_height
        rate = float(metric["pass_rate"])
        bar_width = int(rate * 600)
        name = str(metric["configuration"])
        bars.append(f'<text x="20" y="{y + 18}" fill="#dce7e5" font-size="14">{name}</text>')
        bars.append(f'<rect x="250" y="{y}" width="600" height="24" rx="4" fill="#263633"/>')
        bars.append(
            f'<rect x="250" y="{y}" width="{bar_width}" height="24" rx="4" fill="#b9f36a"/>'
        )
        bars.append(f'<text x="865" y="{y + 18}" fill="#dce7e5" font-size="14">{rate:.1%}</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#091211"/>'
        '<text x="20" y="28" fill="#b9f36a" font-size="18" font-weight="bold">'
        "AgentQA-Bench pass rate</text>" + "".join(bars) + "</svg>\n"
    )
    _write_text(path, svg)


async def run_benchmark(
    *,
    benchmark: BenchmarkConfig,
    output: Path,
    configuration_names: set[str] | None,
    limit: int | None,
    provider: JudgeProvider | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in benchmark.configurations:
        if configuration_names is not None and profile.name not in configuration_names:
            continue
        for suite in benchmark.suites:
            documents = _profile_documents(
                _read_documents(suite.cases), profile.evaluator_types, limit=limit
            )
            if profile.replay_mode == "replay":
                current = await _replay_run(
                    benchmark=benchmark,
                    suite=suite,
                    profile=profile,
                    documents=documents,
                    output=output,
                    provider=provider,
                )
            else:
                current = await _standard_run(
                    benchmark=benchmark,
                    suite=suite,
                    profile=profile,
                    documents=documents,
                    output=output,
                    provider=provider,
                )
            rows.extend(current)
            passed = sum(bool(row["passed"]) for row in current)
            print(f"{profile.name}/{suite.name}: {passed}/{len(current)} passed")
    return rows


def _provider(name: str, benchmark: BenchmarkConfig) -> JudgeProvider | None:
    if name == "none":
        return None
    load_judge_environment(benchmark.judge_env_file)
    return DeepSeekJudgeProvider.from_environment()


def _prepare_output(path: Path) -> Path:
    output = path.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "failures").mkdir(exist_ok=True)
    (output / "charts").mkdir(exist_ok=True)
    return output


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("benchmark/config.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-provider", choices=("none", "deepseek"), default="none")
    parser.add_argument("--configurations", help="Comma-separated configuration names")
    parser.add_argument("--limit", type=int, help="Limit cases per suite for a smoke run")
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be positive")
    benchmark = load_config(arguments.config)
    output = _prepare_output(arguments.output)
    requested = (
        None
        if arguments.configurations is None
        else {value.strip() for value in arguments.configurations.split(",") if value.strip()}
    )
    known = {profile.name for profile in benchmark.configurations}
    if requested is not None and not requested <= known:
        parser.error("unknown configuration(s): " + ", ".join(sorted(requested - known)))
    provider = _provider(arguments.judge_provider, benchmark)
    try:
        rows = asyncio.run(
            run_benchmark(
                benchmark=benchmark,
                output=output,
                configuration_names=requested,
                limit=arguments.limit,
                provider=provider,
            )
        )
    except BenchmarkExecutionError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    raw_path = output / "raw_results.jsonl"
    _write_text(
        raw_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    metrics = _metrics(rows)
    summary = {
        "benchmarkVersion": benchmark.benchmark_version,
        "generatedAt": datetime.now(UTC).isoformat(),
        "judgeProvider": arguments.judge_provider,
        "resultCount": len(rows),
        "configurations": metrics,
        "categoryCounts": dict(Counter(row["labels"]["category"] for row in rows)),
        "rawResultsHash": _file_hash(raw_path),
    }
    _write_text(
        output / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    metric_fields = list(metrics[0]) if metrics else ["configuration"]
    _write_csv(output / "metrics.csv", metrics, metric_fields)
    mutation_rows = _mutation_rows(rows)
    _write_csv(
        output / "mutation_matrix.csv",
        mutation_rows,
        [
            "configuration",
            "suite",
            "test_case_id",
            "mutation_id",
            "operation",
            "triggered",
            "detected",
        ],
    )
    _write_chart(output / "charts" / "pass-rate.svg", metrics)
    manifest_path = benchmark.root / "benchmark" / "manifest.json"
    try:
        package_version = importlib.metadata.version("agentqakit")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source-tree"
    environment = {
        "agentqakitVersion": package_version,
        "benchmarkManifestHash": _file_hash(manifest_path),
        "configHash": _file_hash(benchmark.source),
        "judgeConfigured": provider is not None,
        "judgeProvider": None if provider is None else provider.name,
        "judgeModel": None if provider is None else provider.model,
        "limitPerSuite": arguments.limit,
        "platform": platform.platform(),
        "python": sys.version,
        "randomSeed": benchmark.random_seed,
        "selectedConfigurations": sorted(requested or known),
    }
    _write_text(
        output / "environment.json",
        json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
