"""CLI command for comparing two completed run artifact directories."""

from pathlib import Path
from typing import Annotated

import typer

from agentqa.comparison import (
    ComparisonError,
    compare_runs,
    load_run_artifact,
    write_report,
)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def compare(
    baseline: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Completed baseline artifacts/runs/<run-id> directory.",
        ),
    ],
    candidate: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Completed candidate artifacts/runs/<run-id> directory.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", resolve_path=True, help="Write checksummed JSON report."),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Print the full comparison report as JSON."),
    ] = False,
) -> None:
    """Compare two verified run artifacts by test_case_id."""
    try:
        report = compare_runs(load_run_artifact(baseline), load_run_artifact(candidate))
        written = None if output is None else write_report(output, report)
    except (ComparisonError, OSError) as exc:
        code = getattr(exc, "code", "COMPARISON_IO_ERROR")
        message = getattr(exc, "message", str(exc))
        typer.echo(f"ERROR {code}: {message}", err=True)
        raise typer.Exit(code=2) from None

    if output_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(f"Baseline: {report.baseline_run_id}")
        typer.echo(f"Candidate: {report.candidate_run_id}")
        typer.echo(
            "Pass rate: "
            f"{_percent(report.baseline.pass_rate)} -> "
            f"{_percent(report.candidate_effective_pass_rate)} effective"
        )
        typer.echo(
            "Cases: "
            f"{report.counts.new_regressions} regressions, "
            f"{report.counts.fixed} fixed, "
            f"{report.counts.persistent_failures} persistent failures, "
            f"{report.counts.new} new, {report.counts.removed} removed"
        )
        for case in report.cases:
            if case.classification != "STABLE":
                typer.echo(f"{case.classification} {case.test_case_id}")
    if written is not None:
        typer.echo(f"Report: {written}")
