"""One shared path from validated spec files to TestRunner execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from agentqa.errors import AgentQAError
from agentqa.evaluators import JudgeProvider, built_in_evaluators
from agentqa.models import AgentTestSpec, MutationSpec, RunResult, TargetSpec
from agentqa.runner import RunnerOptions, TestRunner
from agentqa.specs import load_specs
from agentqa.targets import create_target_adapter
from agentqa.tracing import InMemoryTraceStore, LocalOTLPReceiver, TraceOptions


class RunServiceError(AgentQAError):
    """A valid request that cannot be assembled into one test run."""

    def __init__(self, message: str, *, code: str = "RUN_SERVICE_ERROR") -> None:
        super().__init__(message, code=code)


@dataclass(frozen=True, slots=True)
class RunExecutionConfig:
    """File-backed execution request shared by CLI and local API."""

    tests_path: Path
    target_path: Path | None = None
    mutations_path: Path | None = None
    artifact_directory: Path = Path("artifacts")
    random_seed: int = 0
    trace_receiver: bool = True
    otlp_port: int = 4318
    otlp_endpoint: str | None = None
    trace_flush_timeout_seconds: float = 0.25
    capture_trace_content: bool = False
    replay_mode: Literal["off", "record", "replay"] = "record"
    replay_cassette: Path | None = None
    allow_live_fallback: bool = False
    replay_volatile_fields: tuple[str, ...] = ()
    generate_failure_bundles: bool = True
    original_command: str | None = None
    run_id: str | None = None
    judge_provider: JudgeProvider | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.otlp_port <= 65_535:
            raise ValueError("otlp_port must be between 0 and 65535")
        if self.trace_receiver and self.otlp_endpoint is not None:
            raise ValueError("local trace receiver and explicit OTLP endpoint are exclusive")


def _load_inputs(
    config: RunExecutionConfig,
) -> tuple[list[AgentTestSpec], TargetSpec, dict[str, MutationSpec]]:
    loaded_tests = load_specs(config.tests_path)
    tests = [item.spec for item in loaded_tests if isinstance(item.spec, AgentTestSpec)]
    target_documents = (
        loaded_tests if config.target_path is None else load_specs(config.target_path)
    )
    targets = [
        item.spec for item in target_documents if getattr(item.spec, "kind", None) == "Target"
    ]
    if not tests:
        raise RunServiceError(
            "tests path does not contain an AgentTest",
            code="RUN_SERVICE_TESTS_MISSING",
        )
    if len(targets) != 1:
        raise RunServiceError(
            "target path must contain exactly one Target",
            code="RUN_SERVICE_TARGET_SELECTION_ERROR",
        )
    target = cast(TargetSpec, targets[0])
    if not all(spec.target_ref in {None, target.metadata.name} for spec in tests):
        raise RunServiceError(
            "one or more AgentTest targetRef values do not match the target",
            code="RUN_SERVICE_TARGET_MISMATCH",
        )
    mutation_documents = (
        loaded_tests if config.mutations_path is None else load_specs(config.mutations_path)
    )
    mutation_specs = [
        item.spec for item in mutation_documents if isinstance(item.spec, MutationSpec)
    ]
    mutation_names = [spec.metadata.name for spec in mutation_specs]
    if len(mutation_names) != len(set(mutation_names)):
        raise RunServiceError(
            "mutation names must be unique",
            code="RUN_SERVICE_DUPLICATE_MUTATION",
        )
    mutations = {spec.metadata.name: spec for spec in mutation_specs}
    missing = sorted(
        {
            spec.mutation
            for spec in tests
            if spec.mutation is not None and spec.mutation not in mutations
        }
    )
    if missing:
        raise RunServiceError(
            "AgentTest references unknown mutation(s): " + ", ".join(missing),
            code="RUN_SERVICE_MUTATION_MISSING",
        )
    return tests, target, mutations


async def execute_run(config: RunExecutionConfig) -> RunResult:
    """Load specs, configure tracing, and execute one run through the core runner."""
    tests, target, mutations = _load_inputs(config)
    adapter = create_target_adapter(target)
    trace_options = TraceOptions(
        flush_timeout_seconds=config.trace_flush_timeout_seconds,
        capture_content=config.capture_trace_content,
    )
    trace_store = InMemoryTraceStore(max_spans_per_trace=trace_options.max_spans_per_trace)
    receiver: LocalOTLPReceiver | None = None
    endpoint = config.otlp_endpoint
    if config.trace_receiver:
        try:
            receiver = LocalOTLPReceiver(
                trace_store,
                port=config.otlp_port,
                options=trace_options,
            )
            receiver.start()
            endpoint = receiver.endpoint
        except OSError as exc:
            raise RunServiceError(
                f"could not start local OTLP receiver: {type(exc).__name__}",
                code="RUN_SERVICE_OTLP_ERROR",
            ) from None
    try:
        runner = TestRunner(
            adapter,
            built_in_evaluators(config.judge_provider),
            options=RunnerOptions(
                artifact_directory=config.artifact_directory,
                random_seed=config.random_seed,
                trace_flush_timeout_seconds=config.trace_flush_timeout_seconds,
                otlp_endpoint=endpoint,
                replay_mode=config.replay_mode,
                replay_cassette=config.replay_cassette,
                allow_live_fallback=config.allow_live_fallback,
                replay_volatile_fields=config.replay_volatile_fields,
                generate_failure_bundles=config.generate_failure_bundles,
                original_command=config.original_command,
                run_id=config.run_id,
            ),
            trace_store=trace_store,
            trace_options=trace_options,
            mutations=mutations,
        )
        return await runner.run(tests, target)
    finally:
        if receiver is not None:
            receiver.stop()
