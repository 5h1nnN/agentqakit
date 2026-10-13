"""Commands for two-phase, manifest-driven AI test generation."""

import asyncio
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import typer

from agentqa.application import GenerationError, GenerationService
from agentqa.evaluators import (
    DeepSeekConfigurationError,
    DeepSeekProvider,
    load_llm_environment,
)
from agentqa.generation import (
    load_generation_plan,
    write_generation_plan,
    write_generation_suite,
)
from agentqa.models import AgentManifest, GenerationGoal, MutationSpec
from agentqa.specs import SpecLoadError, load_spec_file, load_specs

_Spec = TypeVar("_Spec")

app = typer.Typer(help="Generate test plans and candidate AgentTest specs.", no_args_is_help=True)


def _single_spec(path: Path, expected: type[_Spec]) -> _Spec:
    matches = [item.spec for item in load_spec_file(path) if isinstance(item.spec, expected)]
    if len(matches) != 1:
        raise GenerationError(
            f"{path.name} must contain exactly one {expected.__name__}",
            code="GENERATION_SPEC_SELECTION_ERROR",
        )
    return matches[0]


def _provider(
    provider: Literal["deepseek"],
    environment_file: Path,
) -> DeepSeekProvider:
    if provider != "deepseek":
        raise GenerationError("unsupported generation provider")
    load_llm_environment(environment_file)
    return DeepSeekProvider.from_environment()


def _print_spec_error(exc: SpecLoadError) -> None:
    for issue in exc.issues:
        typer.echo(f"ERROR {issue.format()}", err=True)


@app.command("plan")
def generate_plan(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="AgentManifest v1alpha2 used as generation evidence.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            resolve_path=True,
            help="New JSON plan path inside the workspace.",
        ),
    ],
    goal: Annotated[
        Path | None,
        typer.Option(
            "--goal",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional GenerationGoal. A bounded default is used when omitted.",
        ),
    ] = None,
    provider: Annotated[
        Literal["deepseek"],
        typer.Option("--provider", help="Structured LLM provider."),
    ] = "deepseek",
    environment_file: Annotated[
        Path,
        typer.Option(
            "--llm-env-file",
            resolve_path=True,
            help="Private dotenv file containing structured LLM settings.",
        ),
    ] = Path(".env.local"),
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Workspace boundary for generated output.",
        ),
    ] = Path("."),
) -> None:
    """Generate and persist a capability-risk-scenario plan."""
    try:
        manifest_spec = _single_spec(manifest, AgentManifest)
        goal_spec = None if goal is None else _single_spec(goal, GenerationGoal)
        service = GenerationService(_provider(provider, environment_file))
        plan = asyncio.run(service.generate_plan(manifest_spec, goal_spec))
        destination = write_generation_plan(plan, output, workspace=workspace)
    except SpecLoadError as exc:
        _print_spec_error(exc)
        raise typer.Exit(code=2) from None
    except (DeepSeekConfigurationError, GenerationError, OSError) as exc:
        typer.echo(f"ERROR {getattr(exc, 'message', str(exc))}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(
        f"Generated plan {plan.generation_id}: {len(plan.items)} item(s), "
        f"{len(plan.questions)} question(s)"
    )
    typer.echo(f"Plan: {destination}")


@app.command("tests")
def generate_tests(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="AgentManifest used to verify all generated references.",
        ),
    ],
    plan_path: Annotated[
        Path,
        typer.Option(
            "--plan",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Checksummed plan produced by `agentqa generate plan`.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            resolve_path=True,
            help="Absent or empty output directory inside the workspace.",
        ),
    ],
    mutations: Annotated[
        Path | None,
        typer.Option(
            "--mutations",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Mutation specs explicitly available to generated tests.",
        ),
    ] = None,
    provider: Annotated[
        Literal["deepseek"],
        typer.Option("--provider", help="Structured LLM provider."),
    ] = "deepseek",
    environment_file: Annotated[
        Path,
        typer.Option(
            "--llm-env-file",
            resolve_path=True,
            help="Private dotenv file containing structured LLM settings.",
        ),
    ] = Path(".env.local"),
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Workspace boundary for generated output.",
        ),
    ] = Path("."),
) -> None:
    """Generate validated candidate AgentTest YAML and a coverage report."""
    try:
        manifest_spec = _single_spec(manifest, AgentManifest)
        plan = load_generation_plan(plan_path)
        mutation_specs = (
            []
            if mutations is None
            else [
                item.spec for item in load_specs(mutations) if isinstance(item.spec, MutationSpec)
            ]
        )
        service = GenerationService(_provider(provider, environment_file))
        report = asyncio.run(service.generate_tests(manifest_spec, plan, mutation_specs))
        paths = write_generation_suite(report, output, workspace=workspace)
    except SpecLoadError as exc:
        _print_spec_error(exc)
        raise typer.Exit(code=2) from None
    except (DeepSeekConfigurationError, GenerationError, OSError) as exc:
        typer.echo(f"ERROR {getattr(exc, 'message', str(exc))}", err=True)
        raise typer.Exit(code=2) from None
    review_required = sum(case.review_status == "REVIEW_REQUIRED" for case in report.cases)
    typer.echo(
        f"Generated {len(report.cases)} test(s): {review_required} require review, "
        f"{len(report.gaps)} coverage gap(s)"
    )
    typer.echo(f"Artifacts: {paths[-1].parent}")


__all__ = ["app"]
