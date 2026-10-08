"""Commands for validating and eventually running agent tests."""

from pathlib import Path
from typing import Annotated

import typer

from agentqa.specs import SpecLoadError, load_specs

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
