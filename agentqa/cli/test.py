"""Commands for validating and running agent tests."""

import asyncio
from pathlib import Path
from typing import Annotated, cast

import typer

from agentqa.models import AgentTestSpec, TargetSpec
from agentqa.runner import RunnerOptions, TestRunner
from agentqa.specs import SpecLoadError, load_specs
from agentqa.targets import TargetConfigurationError, create_target_adapter

app = typer.Typer(help="Validate and run AgentTest specifications.", no_args_is_help=True)


@app.command("validate")
def validate(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="A YAML spec file or directory containing YAML specs.",
        ),
    ],
) -> None:
    """Validate YAML specs without executing an agent."""
    try:
        loaded = load_specs(path)
    except SpecLoadError as exc:
        for issue in exc.issues:
            typer.echo(f"ERROR {issue.format()}", err=True)
        typer.echo(f"Validation failed: {len(exc.issues)} issue(s).", err=True)
        raise typer.Exit(code=1) from None

    for item in loaded:
        name = item.spec.metadata.name
        typer.echo(
            f"OK {item.source}#document-{item.document_index} "
            f"{item.spec.kind}/{name} {item.spec_hash}"
        )
    typer.echo(f"Validated {len(loaded)} specification(s).")


def _load_for_run(
    tests_path: Path,
    target_path: Path | None,
) -> tuple[list[AgentTestSpec], TargetSpec]:
    loaded_tests = load_specs(tests_path)
    tests = [item.spec for item in loaded_tests if isinstance(item.spec, AgentTestSpec)]
    target_documents = loaded_tests if target_path is None else load_specs(target_path)
    targets = [
        item.spec for item in target_documents if getattr(item.spec, "kind", None) == "Target"
    ]
    if not tests:
        raise typer.BadParameter("tests path does not contain an AgentTest")
    if len(targets) != 1:
        raise typer.BadParameter("select a path containing exactly one Target with --target")
    target = targets[0]
    if not all(spec.target_ref in {None, target.metadata.name} for spec in tests):
        raise typer.BadParameter("one or more AgentTest targetRef values do not match the target")
    return tests, cast(TargetSpec, target)


@app.command("run")
def run_tests(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="A YAML AgentTest file or directory.",
        ),
    ],
    target: Annotated[
        Path | None,
        typer.Option(
            "--target",
            "-t",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="A path containing exactly one Target. Defaults to PATH.",
        ),
    ] = None,
    artifacts: Annotated[
        Path,
        typer.Option(
            "--artifacts",
            resolve_path=True,
            help="Directory in which run artifacts are written.",
        ),
    ] = Path("artifacts"),
    seed: Annotated[int, typer.Option(help="Deterministic base random seed.")] = 0,
) -> None:
    """Execute AgentTest specifications against a configured target."""
    try:
        tests, target_spec = _load_for_run(path, target)
        adapter = create_target_adapter(target_spec)
    except SpecLoadError as exc:
        for issue in exc.issues:
            typer.echo(f"ERROR {issue.format()}", err=True)
        raise typer.Exit(code=2) from None
    except (TargetConfigurationError, typer.BadParameter) as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(code=2) from None

    runner = TestRunner(
        adapter,
        {},
        options=RunnerOptions(
            artifact_directory=artifacts,
            random_seed=seed,
        ),
    )
    result = asyncio.run(runner.run(tests, target_spec))
    typer.echo(
        f"Run {result.run_id}: {result.status} "
        f"({result.summary.passed} passed, {result.summary.failed} failed, "
        f"{result.summary.incomplete} incomplete)"
    )
    typer.echo(f"Artifacts: {(artifacts / 'runs' / result.run_id).resolve()}")
    if result.error is not None:
        typer.echo(f"ERROR {result.error.code}: {result.error.message}", err=True)
    if result.exit_code != 0:
        raise typer.Exit(code=result.exit_code)
