"""CLI commands for declarative quality-gate evaluation."""

from pathlib import Path
from typing import Annotated

import typer

from agentqa.comparison import (
    ComparisonError,
    compare_runs,
    load_run_artifact,
    write_report,
)
from agentqa.gates import GateError, evaluate_gate
from agentqa.models import QualityGateSpec
from agentqa.specs import SpecLoadError, load_spec_file

app = typer.Typer(help="Evaluate declarative quality gates.", no_args_is_help=True)


def _load_gate(path: Path) -> QualityGateSpec:
    loaded = load_spec_file(path)
    gates = [item.spec for item in loaded if isinstance(item.spec, QualityGateSpec)]
    if len(loaded) != 1 or len(gates) != 1:
        raise GateError(
            "gate file must contain exactly one QualityGate document",
            code="GATE_SPEC_ERROR",
        )
    return gates[0]


@app.command("evaluate")
def evaluate(
    gate: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="QualityGate YAML specification.",
        ),
    ],
    result: Annotated[
        Path,
        typer.Option(
            "--result",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Completed candidate artifacts/runs/<run-id> directory.",
        ),
    ],
    baseline: Annotated[
        Path | None,
        typer.Option(
            "--baseline",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Completed baseline run required by relative rules.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", resolve_path=True, help="Write checksummed gate JSON."),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Print the full gate result as JSON."),
    ] = False,
) -> None:
    """Evaluate a QualityGate and exit 0 (pass), 1 (gate fail), or 2 (error)."""
    try:
        gate_spec = _load_gate(gate)
        candidate = load_run_artifact(result)
        comparison = (
            None if baseline is None else compare_runs(load_run_artifact(baseline), candidate)
        )
        gate_result = evaluate_gate(gate_spec, candidate, comparison)
        written = None if output is None else write_report(output, gate_result)
    except SpecLoadError as exc:
        for issue in exc.issues:
            typer.echo(f"ERROR {issue.format()}", err=True)
        raise typer.Exit(code=2) from None
    except (ComparisonError, GateError, OSError) as exc:
        code = getattr(exc, "code", "GATE_IO_ERROR")
        message = getattr(exc, "message", str(exc))
        typer.echo(f"ERROR {code}: {message}", err=True)
        raise typer.Exit(code=2) from None

    if output_json:
        typer.echo(gate_result.model_dump_json(indent=2))
    else:
        for rule in gate_result.rules:
            status = "PASS" if rule.passed else "FAIL"
            typer.echo(
                f"{status} {rule.name}: actual={rule.actual!r} "
                f"threshold={rule.expected!r} - {rule.message}"
            )
        typer.echo("Quality gate: PASS" if gate_result.passed else "Quality gate: FAIL")
    if written is not None:
        typer.echo(f"Report: {written}")
    if gate_result.exit_code:
        raise typer.Exit(code=gate_result.exit_code)
