"""Verify and execute self-contained failure reproduction bundles."""

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from pydantic import ValidationError

from agentqa.evaluators import built_in_evaluators
from agentqa.models import AgentManifest, AgentTestSpec, MutationSpec, TargetSpec, TestResult
from agentqa.replay import CassetteFormatError, ReplayError
from agentqa.replay.bundle import extract_failure_bundle
from agentqa.runner import RunnerOptions, TestRunner
from agentqa.specs import SpecLoadError, load_spec_file
from agentqa.targets import TargetConfigurationError, create_target_adapter

_T = TypeVar("_T")


def _single_spec(path: Path, expected: type[_T]) -> _T:
    matches = [item.spec for item in load_spec_file(path) if isinstance(item.spec, expected)]
    if len(matches) != 1:
        raise CassetteFormatError(
            f"failure bundle must contain exactly one {expected.__name__} in {path.name}"
        )
    return matches[0]


def _single_target(path: Path) -> TargetSpec:
    matches = [item.spec for item in load_spec_file(path) if item.spec.kind == "Target"]
    if len(matches) != 1:
        raise CassetteFormatError("failure bundle must contain exactly one Target in target.yaml")
    return matches[0]


def _signature(result: TestResult) -> tuple[Any, ...]:
    return (
        result.passed,
        result.target_result.status,
        tuple(sorted(item.type for item in result.violations)),
        tuple(sorted((item.evaluator_type, item.status) for item in result.evaluations)),
        tuple(
            sorted(
                (item.mutation_id, item.operation_type, item.triggered)
                for item in result.mutation_applications
            )
        ),
    )


def run_replay(
    bundle: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="A failure-<trial-id>.zip reproduction bundle.",
        ),
    ],
    artifacts: Annotated[
        Path,
        typer.Option(
            "--artifacts",
            resolve_path=True,
            help="Directory in which replay-run artifacts are written.",
        ),
    ] = Path("artifacts"),
    allow_live_fallback: Annotated[
        bool,
        typer.Option(
            help="Allow cassette misses to access live dependencies and record a delta.",
        ),
    ] = False,
) -> None:
    """Verify BUNDLE checksums, replay one trial, and compare its failure signature."""
    try:
        with tempfile.TemporaryDirectory(prefix="agentqa-replay-") as temporary:
            root = Path(temporary)
            manifest_data = extract_failure_bundle(bundle, root)
            test_spec = _single_spec(root / "test.yaml", AgentTestSpec)
            target_spec = _single_target(root / "target.yaml")
            mutation_spec = (
                _single_spec(root / "mutation.yaml", MutationSpec)
                if bool(manifest_data.get("has_mutation"))
                else None
            )
            agent_manifest = (
                _single_spec(root / "agent-manifest.yaml", AgentManifest)
                if bool(manifest_data.get("has_agent_manifest"))
                else None
            )
            original = TestResult.model_validate_json(
                (root / "result.json").read_text(encoding="utf-8")
            )
            if original.passed:
                raise CassetteFormatError("failure bundle contains a passing result")
            random_seed = manifest_data.get("random_seed")
            trial_seed = manifest_data.get("trial_seed")
            if not isinstance(random_seed, int) or not isinstance(trial_seed, int):
                raise CassetteFormatError("failure bundle seed metadata is invalid")
            mutations = (
                {} if mutation_spec is None else {mutation_spec.metadata.name: mutation_spec}
            )
            runner = TestRunner(
                create_target_adapter(target_spec),
                built_in_evaluators(),
                options=RunnerOptions(
                    artifact_directory=artifacts,
                    random_seed=random_seed,
                    replay_mode="replay",
                    replay_cassette=root / "cassette.jsonl",
                    allow_live_fallback=allow_live_fallback,
                    replay_trial_seed=trial_seed,
                    generate_failure_bundles=False,
                    original_command=f"agentqa replay {bundle}",
                ),
                mutations=mutations,
            )
            replayed_run = asyncio.run(
                runner.run([test_spec], target_spec, manifest=agent_manifest)
            )
            if replayed_run.status != "COMPLETED" or len(replayed_run.results) != 1:
                detail = (
                    replayed_run.error.code
                    if replayed_run.error is not None
                    else replayed_run.status
                )
                typer.echo(f"ERROR replay infrastructure failed: {detail}", err=True)
                raise typer.Exit(code=2)
            replayed = replayed_run.results[0]
            if _signature(replayed) != _signature(original):
                typer.echo(
                    "Replay completed, but the recorded failure signature was not reproduced.",
                    err=True,
                )
                typer.echo(f"Replay artifacts: {artifacts / 'runs' / replayed_run.run_id}")
                raise typer.Exit(code=1)
            hit_text = "n/a"
            if replayed.replay is not None and replayed.replay.hit_rate is not None:
                hit_text = f"{replayed.replay.hit_rate:.1%}"
            typer.echo(
                f"Reproduced failure for {replayed.test_case_id} (cassette hit rate: {hit_text})."
            )
            typer.echo(f"Replay artifacts: {artifacts / 'runs' / replayed_run.run_id}")
    except SpecLoadError as exc:
        for issue in exc.issues:
            typer.echo(f"ERROR {issue.format()}", err=True)
        raise typer.Exit(code=2) from None
    except (CassetteFormatError, ReplayError, TargetConfigurationError, ValidationError) as exc:
        typer.echo(f"ERROR invalid reproduction bundle: {type(exc).__name__}", err=True)
        raise typer.Exit(code=2) from None
    except OSError as exc:
        typer.echo(f"ERROR could not read reproduction bundle: {type(exc).__name__}", err=True)
        raise typer.Exit(code=2) from None
