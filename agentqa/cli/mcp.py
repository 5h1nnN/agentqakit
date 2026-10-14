"""Optional local MCP server command."""

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from agentqa.evaluators import (
    DeepSeekConfigurationError,
    DeepSeekProvider,
    load_llm_environment,
)

app = typer.Typer(help="Collaborate with Coding Agents through local MCP.", no_args_is_help=True)


def _provider(environment_file: Path) -> DeepSeekProvider | None:
    if environment_file.exists():
        load_llm_environment(environment_file)
    configured = any(
        os.environ.get(name)
        for name in (
            "AGENTQA_LLM_API_KEY",
            "AGENTQA_LLM_MODEL",
            "AGENTQA_JUDGE_API_KEY",
            "AGENTQA_JUDGE_MODEL",
        )
    )
    return DeepSeekProvider.from_environment() if configured else None


@app.command("serve")
def serve(
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Fixed workspace boundary for all MCP resources and tools.",
        ),
    ] = Path("."),
    manifest_path: Annotated[
        str,
        typer.Option(
            "--manifest",
            help="Workspace-relative manifest exposed as agentqa://project/manifest.",
        ),
    ] = "agent-manifest.yaml",
    environment_file: Annotated[
        Path | None,
        typer.Option(
            "--llm-env-file",
            help="Private dotenv file; defaults to <workspace>/.env.local when present.",
        ),
    ] = None,
) -> None:
    """Serve AgentQAKit over stdio without mixing logs into protocol output."""
    try:
        from agentqa.mcp.server import create_mcp_server
    except ModuleNotFoundError as exc:
        if exc.name == "mcp" or (exc.name is not None and exc.name.startswith("mcp.")):
            typer.echo(
                'ERROR MCP support is not installed; run pip install "agentqakit[mcp]"',
                err=True,
            )
            raise typer.Exit(code=2) from None
        raise
    source = environment_file or (workspace / ".env.local")
    try:
        provider = _provider(source.resolve())
        server = create_mcp_server(
            workspace,
            manifest_path=manifest_path,
            provider=provider,
        )
    except DeepSeekConfigurationError as exc:
        typer.echo(f"ERROR {exc.code}: {exc.message}", err=True)
        raise typer.Exit(code=2) from None
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server.run("stdio")


__all__ = ["app"]
