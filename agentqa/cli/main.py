"""Top-level AgentQAKit command-line application."""

from typing import Annotated

import typer

from agentqa import __version__
from agentqa.cli.instrument import instrument
from agentqa.cli.mutation import app as mutation_app
from agentqa.cli.test import app as test_app

app = typer.Typer(
    name="agentqa",
    help="Fault-injection and regression testing for AI agents.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(test_app, name="test")
app.add_typer(mutation_app, name="mutation")
app.command("instrument")(instrument)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"AgentQAKit {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the AgentQAKit version and exit.",
        ),
    ] = None,
) -> None:
    """Run AgentQAKit commands."""
