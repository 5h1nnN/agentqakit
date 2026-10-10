"""Commands for validating and running agent tests."""

import asyncio
import shlex
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from agentqa.application import RunExecutionConfig, RunServiceError, execute_run
from agentqa.specs import SpecLoadError, load_specs
from agentqa.targets import TargetConfigurationError

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


@app.command("run")
def run_tests(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="A YAML AgentTest file or directory.",
        ),
    ],
    target: Annotated[
        Path | None,
        typer.Option(
            "--target",
            "-t",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="A path containing exactly one Target. Defaults to PATH.",
        ),
    ] = None,
    mutations: Annotated[
        Path | None,
        typer.Option(
            "--mutations",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help=(
                "A path containing Mutation specs. Defaults to PATH, allowing tests, "
                "target, and mutations in one multi-document YAML file."
            ),
        ),
    ] = None,
    artifacts: Annotated[
        Path,
        typer.Option(
            "--artifacts",
            resolve_path=True,
            help="Directory in which run artifacts are written.",
        ),
    ] = Path("artifacts"),
    seed: Annotated[int, typer.Option(help="Deterministic base random seed.")] = 0,
    otlp_port: Annotated[
        int,
        typer.Option(min=0, max=65_535, help="Loopback OTLP/HTTP receiver port."),
    ] = 4318,
    trace_receiver: Annotated[
        bool,
        typer.Option(
            "--trace-receiver/--no-trace-receiver",
            help="Collect standard OTLP spans from instrumented target processes.",
        ),
    ] = True,
    trace_flush_timeout: Annotated[
        float,
        typer.Option(min=0.01, help="Seconds to await spans after a target response."),
    ] = 0.25,
    capture_trace_content: Annotated[
        bool,
        typer.Option(help="Persist redacted, size-limited span input and output content."),
    ] = False,
    replay_mode: Annotated[
        str,
        typer.Option(
            "--replay-mode",
            help="Dependency interaction policy: off, record, or replay.",
        ),
    ] = "record",
    replay_cassette: Annotated[
        Path | None,
        typer.Option(
            "--replay-cassette",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Source JSONL cassette required by replay mode.",
        ),
    ] = None,
    allow_live_fallback: Annotated[
        bool,
        typer.Option(
            help="Allow a replay miss to call the live dependency and record a delta.",
        ),
    ] = False,
    volatile_field: Annotated[
        list[str] | None,
        typer.Option(
            "--volatile-field",
            help="Argument field path to remove before cassette matching; repeatable.",
        ),
    ] = None,
    failure_bundles: Annotated[
        bool,
        typer.Option(
            "--failure-bundles/--no-failure-bundles",
            help="Create a verified reproduction ZIP for each failed trial.",
        ),
    ] = True,
) -> None:
    """Execute AgentTest specifications against a configured target."""
    if replay_mode not in {"off", "record", "replay"}:
        raise typer.BadParameter(
            "replay mode must be off, record, or replay",
            param_hint="--replay-mode",
        )
    if replay_mode == "replay" and replay_cassette is None:
        raise typer.BadParameter(
            "replay mode requires --replay-cassette",
            param_hint="--replay-cassette",
        )
    if replay_mode != "replay" and allow_live_fallback:
        raise typer.BadParameter(
            "--allow-live-fallback is only valid in replay mode",
            param_hint="--allow-live-fallback",
        )
    command = ["agentqa", "test", "run", str(path)]
    if target is not None:
        command.extend(["--target", str(target)])
    if mutations is not None:
        command.extend(["--mutations", str(mutations)])
    command.extend(
        [
            "--artifacts",
            str(artifacts),
            "--seed",
            str(seed),
            "--otlp-port",
            str(otlp_port),
            "--trace-flush-timeout",
            str(trace_flush_timeout),
            "--replay-mode",
            replay_mode,
        ]
    )
    command.append("--trace-receiver" if trace_receiver else "--no-trace-receiver")
    if capture_trace_content:
        command.append("--capture-trace-content")
    if replay_cassette is not None:
        command.extend(["--replay-cassette", str(replay_cassette)])
    if allow_live_fallback:
        command.append("--allow-live-fallback")
    for field in volatile_field or ():
        command.extend(["--volatile-field", field])
    try:
        result = asyncio.run(
            execute_run(
                RunExecutionConfig(
                    tests_path=path,
                    target_path=target,
                    mutations_path=mutations,
                    artifact_directory=artifacts,
                    random_seed=seed,
                    trace_receiver=trace_receiver,
                    otlp_port=otlp_port,
                    trace_flush_timeout_seconds=trace_flush_timeout,
                    capture_trace_content=capture_trace_content,
                    replay_mode=cast(Literal["off", "record", "replay"], replay_mode),
                    replay_cassette=replay_cassette,
                    allow_live_fallback=allow_live_fallback,
                    replay_volatile_fields=tuple(volatile_field or ()),
                    generate_failure_bundles=failure_bundles,
                    original_command=shlex.join(command),
                )
            )
        )
    except SpecLoadError as exc:
        for issue in exc.issues:
            typer.echo(f"ERROR {issue.format()}", err=True)
        raise typer.Exit(code=2) from None
    except (RunServiceError, TargetConfigurationError, ValueError) as exc:
        message = getattr(exc, "message", str(exc))
        typer.echo(f"ERROR {message}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(
        f"Run {result.run_id}: {result.status} "
        f"({result.summary.passed} passed, {result.summary.failed} failed, "
        f"{result.summary.incomplete} incomplete)"
    )
    typer.echo(f"Artifacts: {(artifacts / 'runs' / result.run_id).resolve()}")
    if result.mutation_score is not None:
        score = result.mutation_score.overall
        score_text = "n/a" if score.score is None else f"{score.score:.1%}"
        typer.echo(
            "Mutation score: "
            f"{score_text} ({score.killed} killed, {score.survived} survived, "
            f"{score.not_reached} not reached, {score.invalid} invalid)"
        )
    if result.error is not None:
        typer.echo(f"ERROR {result.error.code}: {result.error.message}", err=True)
    if result.exit_code != 0:
        raise typer.Exit(code=result.exit_code)
