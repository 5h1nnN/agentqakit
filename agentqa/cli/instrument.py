"""Zero-source-change child-process instrumentation command."""

import os
import shutil
import signal
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated, Any

import typer

from agentqa.tracing.instrumentation import instrumentation_install_hint

_FRAMEWORK_INSTRUMENTORS = {
    "openai": {"openai"},
    "langchain": {"langchain"},
    "langgraph": {"langchain", "openai"},
    "httpx": {"httpx"},
}


def _launcher() -> str | None:
    adjacent = Path(sys.executable).with_name("opentelemetry-instrument")
    if adjacent.is_file():
        return str(adjacent)
    return shutil.which("opentelemetry-instrument")


def _installed_instrumentors() -> set[str]:
    return {item.name for item in entry_points(group="opentelemetry_instrumentor")}


def _forward_signal(process: subprocess.Popen[Any], signum: int) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
    else:
        process.send_signal(signum)


def instrument(
    command: Annotated[
        list[str] | None,
        typer.Argument(help="Command to execute after a -- separator."),
    ] = None,
    framework: Annotated[
        list[str] | None,
        typer.Option("--framework", "-f", help="Framework instrumentor to enable."),
    ] = None,
    exporter: Annotated[
        str,
        typer.Option(help="Base OTLP/HTTP endpoint receiving child spans."),
    ] = "http://127.0.0.1:4318",
    capture_content: Annotated[
        bool,
        typer.Option(help="Allow instrumentors to export redacted prompt and response content."),
    ] = False,
) -> None:
    """Run a command under selected OTel auto-instrumentors without rewriting source."""
    if not command:
        typer.echo("ERROR provide a command after --", err=True)
        raise typer.Exit(code=2)
    requested = framework or ["langgraph"]
    unsupported = sorted(set(requested) - _FRAMEWORK_INSTRUMENTORS.keys())
    if unsupported:
        typer.echo(f"ERROR unsupported framework: {', '.join(unsupported)}", err=True)
        raise typer.Exit(code=2)

    installed = _installed_instrumentors()
    enabled = set().union(*(_FRAMEWORK_INSTRUMENTORS[item] for item in requested))
    missing = sorted(name for name in enabled if name not in installed)
    if missing:
        for name in missing:
            typer.echo(f"ERROR {instrumentation_install_hint(name)}", err=True)
        raise typer.Exit(code=2)
    launcher = _launcher()
    if launcher is None:
        typer.echo(
            "ERROR opentelemetry-instrument is unavailable; reinstall agentqakit core dependencies",
            err=True,
        )
        raise typer.Exit(code=2)

    environment = os.environ.copy()
    environment.update(
        {
            "AGENTQA_FRAMEWORKS": ",".join(requested),
            "AGENTQA_CAPTURE_CONTENT": "true" if capture_content else "false",
            "OTEL_EXPORTER_OTLP_ENDPOINT": exporter.rstrip("/"),
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_TRACES_EXPORTER": "otlp",
            "OTEL_SERVICE_NAME": "agentqa-instrumented-target",
            "OTEL_BSP_SCHEDULE_DELAY": "100",
            "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": ",".join(sorted(installed - enabled)),
            "OPENINFERENCE_HIDE_INPUTS": "false" if capture_content else "true",
            "OPENINFERENCE_HIDE_OUTPUTS": "false" if capture_content else "true",
        }
    )
    try:
        child = subprocess.Popen(
            [launcher, *command],
            env=environment,
            start_new_session=True,
        )
    except OSError as exc:
        typer.echo(f"ERROR could not start instrumented command: {type(exc).__name__}", err=True)
        raise typer.Exit(code=2) from None

    previous_handlers: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        del frame
        _forward_signal(child, signum)

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, handler)
    try:
        return_code = child.wait()
    finally:
        for restore_number, previous in previous_handlers.items():
            signal.signal(restore_number, previous)
    if return_code < 0:
        return_code = 128 + abs(return_code)
    if return_code:
        raise typer.Exit(code=return_code)
