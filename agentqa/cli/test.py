"""Commands for validating and running agent tests."""

import asyncio
from pathlib import Path
from typing import Annotated, cast

import typer

from agentqa.evaluators import built_in_evaluators
from agentqa.models import AgentTestSpec, MutationSpec, TargetSpec
from agentqa.runner import RunnerOptions, TestRunner
from agentqa.specs import SpecLoadError, load_specs
from agentqa.targets import TargetConfigurationError, create_target_adapter
from agentqa.tracing import InMemoryTraceStore, LocalOTLPReceiver, TraceOptions

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


def _load_for_run(
    tests_path: Path,
    target_path: Path | None,
    mutations_path: Path | None,
) -> tuple[list[AgentTestSpec], TargetSpec, dict[str, MutationSpec]]:
    loaded_tests = load_specs(tests_path)
    tests = [item.spec for item in loaded_tests if isinstance(item.spec, AgentTestSpec)]
    target_documents = loaded_tests if target_path is None else load_specs(target_path)
    targets = [
        item.spec for item in target_documents if getattr(item.spec, "kind", None) == "Target"
    ]
    if not tests:
        raise typer.BadParameter("tests path does not contain an AgentTest")
    if len(targets) != 1:
        raise typer.BadParameter("select a path containing exactly one Target with --target")
    target = targets[0]
    if not all(spec.target_ref in {None, target.metadata.name} for spec in tests):
        raise typer.BadParameter("one or more AgentTest targetRef values do not match the target")
    mutation_documents = loaded_tests if mutations_path is None else load_specs(mutations_path)
    mutation_specs = [
        item.spec for item in mutation_documents if isinstance(item.spec, MutationSpec)
    ]
    mutation_names = [spec.metadata.name for spec in mutation_specs]
    if len(mutation_names) != len(set(mutation_names)):
        raise typer.BadParameter("mutation names must be unique")
    mutations = {spec.metadata.name: spec for spec in mutation_specs}
    missing = sorted(
        {
            spec.mutation
            for spec in tests
            if spec.mutation is not None and spec.mutation not in mutations
        }
    )
    if missing:
        raise typer.BadParameter("AgentTest references unknown mutation(s): " + ", ".join(missing))
    return tests, cast(TargetSpec, target), mutations


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
) -> None:
    """Execute AgentTest specifications against a configured target."""
    try:
        tests, target_spec, mutation_specs = _load_for_run(path, target, mutations)
        adapter = create_target_adapter(target_spec)
    except SpecLoadError as exc:
        for issue in exc.issues:
            typer.echo(f"ERROR {issue.format()}", err=True)
        raise typer.Exit(code=2) from None
    except (TargetConfigurationError, typer.BadParameter) as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(code=2) from None

    trace_options = TraceOptions(
        flush_timeout_seconds=trace_flush_timeout,
        capture_content=capture_trace_content,
    )
    trace_store = InMemoryTraceStore(max_spans_per_trace=trace_options.max_spans_per_trace)
    receiver: LocalOTLPReceiver | None = None
    endpoint: str | None = None
    if trace_receiver:
        try:
            receiver = LocalOTLPReceiver(trace_store, port=otlp_port, options=trace_options)
            receiver.start()
            endpoint = receiver.endpoint
        except OSError as exc:
            typer.echo(
                f"ERROR could not start local OTLP receiver: {type(exc).__name__}; "
                "choose --otlp-port or use --no-trace-receiver",
                err=True,
            )
            raise typer.Exit(code=2) from None
    try:
        runner = TestRunner(
            adapter,
            built_in_evaluators(),
            options=RunnerOptions(
                artifact_directory=artifacts,
                random_seed=seed,
                trace_flush_timeout_seconds=trace_flush_timeout,
                otlp_endpoint=endpoint,
            ),
            trace_store=trace_store,
            trace_options=trace_options,
            mutations=mutation_specs,
        )
        result = asyncio.run(runner.run(tests, target_spec))
    finally:
        if receiver is not None:
            receiver.stop()
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
