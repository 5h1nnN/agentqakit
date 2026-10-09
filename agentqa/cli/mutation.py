"""Mutation result inspection and fault-proxy commands."""

import json
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer

from agentqa.models import AgentTestSpec, MutationSpec, TestResult
from agentqa.mutations import (
    FaultProxy,
    FaultProxyError,
    FaultProxyOptions,
    calculate_mutation_score,
)
from agentqa.specs import SpecLoadError, load_specs
from agentqa.storage import ArtifactError, ArtifactWriter

app = typer.Typer(
    help="Inspect mutation results and run HTTP fault injection.",
    no_args_is_help=True,
)


def _artifact_reader(run_directory: Path) -> ArtifactWriter:
    resolved = run_directory.resolve()
    if resolved.parent.name != "runs":
        raise ArtifactError("RUN_DIRECTORY must be an artifacts/runs/<run-id> directory")
    reader = ArtifactWriter(resolved.parent.parent, resolved.name)
    if reader.root != resolved:
        raise ArtifactError("RUN_DIRECTORY could not be resolved safely")
    return reader


def _read_document(reader: ArtifactWriter, name: str) -> dict[str, Any]:
    if not reader.verify_checksum(name):
        raise ArtifactError(f"{name} checksum mismatch", code="ARTIFACT_CHECKSUM_ERROR")
    try:
        value = json.loads((reader.root / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ArtifactError(f"{name} is not valid JSON", code="ARTIFACT_FORMAT_ERROR") from None
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must contain a JSON object", code="ARTIFACT_FORMAT_ERROR")
    value.pop("schema_version", None)
    return cast(dict[str, Any], value)


def load_mutation_run(
    run_directory: Path,
) -> tuple[dict[str, MutationSpec], list[AgentTestSpec], list[TestResult]]:
    """Load and verify the evidence required to recompute a mutation score."""
    reader = _artifact_reader(run_directory)
    mutation_document = _read_document(reader, "resolved-mutations.json")
    test_document = _read_document(reader, "resolved-tests.json")
    try:
        mutation_specs = [
            MutationSpec.model_validate(value) for value in mutation_document["mutations"]
        ]
        test_specs = [AgentTestSpec.model_validate(value) for value in test_document["tests"]]
        results = [
            TestResult.model_validate_json(json.dumps(value))
            for value in reader.read_result_records()
        ]
    except (KeyError, TypeError, ValueError):
        raise ArtifactError(
            "mutation run artifacts do not match the expected schema",
            code="ARTIFACT_FORMAT_ERROR",
        ) from None
    names = [spec.metadata.name for spec in mutation_specs]
    if len(names) != len(set(names)):
        raise ArtifactError(
            "resolved mutation names must be unique",
            code="ARTIFACT_FORMAT_ERROR",
        )
    return {spec.metadata.name: spec for spec in mutation_specs}, test_specs, results


@app.command("score")
def score(
    run_directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="A completed artifacts/runs/<run-id> directory.",
        ),
    ],
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Print the full recomputed report as JSON."),
    ] = False,
) -> None:
    """Recompute Mutation Score from checksummed run evidence."""
    try:
        mutations, tests, results = load_mutation_run(run_directory)
        report = calculate_mutation_score(mutations, tests, results)
    except ArtifactError as exc:
        typer.echo(f"ERROR {exc.code}: {exc.message}", err=True)
        raise typer.Exit(code=2) from None

    if output_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    overall = report.overall
    value = "n/a" if overall.score is None else f"{overall.score:.1%}"
    typer.echo(f"Mutation score: {value}")
    typer.echo(
        f"Killed: {overall.killed}  Survived: {overall.survived}  "
        f"Not reached: {overall.not_reached}  Invalid: {overall.invalid}"
    )
    for operation, bucket in report.by_operation.items():
        operation_score = "n/a" if bucket.score is None else f"{bucket.score:.1%}"
        typer.echo(
            f"{operation}: {operation_score} "
            f"(K={bucket.killed}, S={bucket.survived}, "
            f"N={bucket.not_reached}, I={bucket.invalid})"
        )


@app.command("proxy")
def proxy(
    upstream: Annotated[
        str,
        typer.Option(
            "--upstream",
            help="One fixed HTTP(S) upstream base URL; request-supplied hosts are ignored.",
        ),
    ],
    allow_path: Annotated[
        list[str],
        typer.Option(
            "--allow-path",
            help="Exact upstream path to permit; repeat for additional paths.",
        ),
    ],
    route_name: Annotated[
        str,
        typer.Option("--route-name", help="Mutation target.name exposed by this route."),
    ],
    mutations_path: Annotated[
        Path,
        typer.Option(
            "--mutations",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="A path containing HTTP Mutation specs.",
        ),
    ],
    cassette_directory: Annotated[
        Path,
        typer.Option(
            "--cassettes",
            resolve_path=True,
            help="Directory for sanitized response cassettes and application evidence.",
        ),
    ] = Path("artifacts/fault-proxy-cassettes"),
    mode: Annotated[
        str,
        typer.Option(help="record forwards to the fixed upstream; replay never does."),
    ] = "record",
    port: Annotated[
        int,
        typer.Option(min=0, max=65_535, help="Loopback listening port; 0 selects one."),
    ] = 0,
) -> None:
    """Run a loopback-only fixed-upstream HTTP fault proxy."""
    if mode not in {"record", "replay"}:
        raise typer.BadParameter("mode must be record or replay", param_hint="--mode")
    try:
        loaded = load_specs(mutations_path)
        mutation_specs = [
            item.spec
            for item in loaded
            if isinstance(item.spec, MutationSpec) and item.spec.target.kind == "http"
        ]
        names = [spec.metadata.name for spec in mutation_specs]
        if not mutation_specs:
            raise typer.BadParameter("mutation path does not contain an HTTP Mutation")
        if len(names) != len(set(names)):
            raise typer.BadParameter("mutation names must be unique")
        fault_proxy = FaultProxy(
            FaultProxyOptions(
                upstream_base_url=upstream,
                allowed_paths=tuple(allow_path),
                route_name=route_name,
                cassette_directory=cassette_directory,
                mode=cast(Literal["record", "replay"], mode),
                port=port,
            ),
            {spec.metadata.name: spec for spec in mutation_specs},
        )
        endpoint = fault_proxy.start()
    except SpecLoadError as exc:
        for issue in exc.issues:
            typer.echo(f"ERROR {issue.format()}", err=True)
        raise typer.Exit(code=2) from None
    except (FaultProxyError, OSError, typer.BadParameter) as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(code=2) from None

    typer.echo(f"Fault proxy: {endpoint}")
    typer.echo("Press Ctrl-C to stop.")
    try:
        fault_proxy.wait()
    except KeyboardInterrupt:
        pass
    finally:
        fault_proxy.stop()
