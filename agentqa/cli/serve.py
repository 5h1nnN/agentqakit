"""Launch the local FastAPI and optional built React application."""

from pathlib import Path
from typing import Annotated

import typer


def serve(
    host: Annotated[
        str,
        typer.Option(help="Listening host; loopback is the secure default."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(min=1, max=65_535, help="HTTP listening port."),
    ] = 8000,
    workspace: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Root allowed for API spec paths.",
        ),
    ] = Path("."),
    data_directory: Annotated[
        Path,
        typer.Option("--data", resolve_path=True, help="SQLite and server state directory."),
    ] = Path(".agentqa"),
    artifacts: Annotated[
        Path,
        typer.Option(resolve_path=True, help="Directory for API-triggered run artifacts."),
    ] = Path(".agentqa/artifacts"),
    static_directory: Annotated[
        Path | None,
        typer.Option(
            "--static",
            resolve_path=True,
            help="Built React dist directory; omit the option to use apps/web/dist.",
        ),
    ] = None,
    allow_remote: Annotated[
        bool,
        typer.Option(help="Permit a non-loopback host. No authentication is provided."),
    ] = False,
) -> None:
    """Serve the local AgentQAKit API and built web UI."""
    if host not in {"127.0.0.1", "::1", "localhost"} and not allow_remote:
        raise typer.BadParameter(
            "non-loopback hosts require --allow-remote",
            param_hint="--host",
        )
    try:
        import uvicorn

        from agentqa.api import ApiSettings, create_app
    except ImportError:
        typer.echo(
            'ERROR API dependencies are missing; install with pip install ".[api]"',
            err=True,
        )
        raise typer.Exit(code=2) from None

    workspace = workspace.resolve()
    default_static = workspace / "apps" / "web" / "dist"
    settings = ApiSettings(
        workspace_root=workspace,
        artifact_directory=artifacts.resolve(),
        database_path=(data_directory.resolve() / "agentqa.sqlite3"),
        static_directory=(
            default_static if static_directory is None else static_directory.resolve()
        ),
    )
    typer.echo(f"AgentQAKit: http://{host}:{port}")
    typer.echo(f"Workspace: {workspace}")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
