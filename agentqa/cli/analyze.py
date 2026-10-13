"""CLI commands for evidence-grounded AI failure analysis."""

import asyncio
from pathlib import Path
from typing import Annotated, Literal

import typer

from agentqa.analysis.io import load_failure_analysis_input, write_analysis_report
from agentqa.application import AnalysisError, FailureAnalysisService
from agentqa.evaluators import (
    DeepSeekConfigurationError,
    DeepSeekProvider,
    load_llm_environment,
)

app = typer.Typer(
    help="Analyze failed trials without changing evaluation evidence.", no_args_is_help=True
)


def _provider(provider: Literal["deepseek"], environment_file: Path) -> DeepSeekProvider:
    if provider != "deepseek":
        raise AnalysisError("unsupported analysis provider")
    load_llm_environment(environment_file)
    return DeepSeekProvider.from_environment()


@app.command("failure")
def analyze_failure(
    run_directory: Annotated[
        Path,
        typer.Option(
            "--run",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Completed artifacts/runs/<run-id> directory.",
        ),
    ],
    trial_id: Annotated[
        str,
        typer.Option("--trial-id", help="Failed trial identifier from results.jsonl."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", resolve_path=True, help="New checksummed JSON report path."),
    ],
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
            help="Workspace boundary for analysis output.",
        ),
    ] = Path("."),
) -> None:
    """Analyze one failed trial from verified immutable run evidence."""
    try:
        inputs = load_failure_analysis_input(run_directory, trial_id)
        service = FailureAnalysisService(_provider(provider, environment_file))
        report = asyncio.run(
            service.analyze(
                run_id=inputs.run_id,
                test_spec=inputs.test_spec,
                result=inputs.result,
                trace=inputs.trace,
                manifest=inputs.manifest,
            )
        )
        destination = write_analysis_report(report, output, workspace=workspace)
    except (AnalysisError, DeepSeekConfigurationError, OSError) as exc:
        code = getattr(exc, "code", "ANALYSIS_IO_ERROR")
        message = getattr(exc, "message", str(exc))
        typer.echo(f"ERROR {code}: {message}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(
        f"Analyzed failed trial {report.trial_id}: "
        f"{len(report.root_causes)} root-cause candidate(s), "
        f"{len(report.missing_evidence)} evidence gap(s)"
    )
    typer.echo(f"Report: {destination}")


__all__ = ["app"]
