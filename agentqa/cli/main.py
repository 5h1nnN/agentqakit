"""Top-level AgentQAKit command-line application."""

from typing import Annotated

import typer

from agentqa import __version__
from agentqa.cli.compare import compare
from agentqa.cli.gate import app as gate_app
from agentqa.cli.generate import app as generate_app
from agentqa.cli.instrument import instrument
from agentqa.cli.mutation import app as mutation_app
from agentqa.cli.replay import run_replay
from agentqa.cli.serve import serve
from agentqa.cli.test import app as test_app

app = typer.Typer(
    name="agentqa",
    help="Fault-injection and regression testing for AI agents.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(test_app, name="test")
app.add_typer(mutation_app, name="mutation")
app.add_typer(gate_app, name="gate")
app.add_typer(generate_app, name="generate")
app.command("compare")(compare)
app.command("replay")(run_replay)
app.command("serve")(serve)
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
