# AgentQAKit

AgentQAKit is a local-first testing toolkit for AI agents. It is designed to run
repeatable agent tests, collect execution traces, inject controlled failures,
evaluate results, reproduce failures, and enforce quality gates in CI.

The project is currently being built from the ground up. The repository now
contains the package foundation, strict specifications, sequential runner,
HTTP, isolated Python, and optional LangGraph target adapters, plus bounded
OpenTelemetry trace collection, OTLP/HTTP instrumentation, deterministic tool,
retriever, and HTTP mutations, and recomputable Mutation Score reports.
It also includes deterministic output, tool, trajectory, state, privacy, and
budget evaluators plus an opt-in, provider-neutral LLM judge boundary, canonical
dependency replay, and verified failure reproduction bundles.

## Requirements

- Python 3.11 or newer
- `pip`

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
# Optional OpenAI/LangGraph auto-instrumentation:
python -m pip install -e ".[instrumentation]"
```

Verify the installation:

```bash
agentqa --help
make lint
make test
make e2e
```

Validate the included declarative specifications:

```bash
agentqa test validate examples/specs/
```

Execute a test against one target:

```bash
agentqa test run examples/specs/basic-test.yaml \
  --target examples/specs/http-target.yaml
```

Run tests with mutation specifications and inspect the resulting score:

```bash
agentqa test run tests.yaml --target target.yaml --mutations mutations.yaml
agentqa mutation score artifacts/runs/<run-id>
```

Reproduce a failed trial without calling its recorded wrapped dependencies:

```bash
agentqa replay artifacts/runs/<run-id>/failures/failure-<trial-id>.zip
```

AgentQAKit accepts strict, versioned YAML. Unknown fields and unsupported
evaluator or target types are rejected instead of being ignored. See
[`docs/specifications.md`](docs/specifications.md) for the supported format.

The sequential runner provides an explicit lifecycle, per-trial isolation,
timeouts, cancellation, partial-failure handling, and stability aggregation.
See [`docs/runner.md`](docs/runner.md). Target construction, security defaults,
subprocess isolation, and LangGraph observation are documented in
[`docs/targets.md`](docs/targets.md). Trace completeness, privacy limits, the
local OTLP receiver, and zero-source-change instrumentation are documented in
[`docs/tracing.md`](docs/tracing.md). Mutation matching, runtime wrappers, the
fixed-upstream fault proxy, evidence, and score semantics are documented in
[`docs/mutations.md`](docs/mutations.md). Evaluator evidence requirements,
status semantics, and Judge integration are documented in
[`docs/evaluators.md`](docs/evaluators.md). Cassette matching, privacy,
strict replay, and verified failure bundles are documented in
[`docs/replay.md`](docs/replay.md).

## Repository layout

- `agentqa/`: importable Python package and CLI
- `apps/`: API and web applications
- `benchmark/`: open benchmark data and configuration
- `examples/`: runnable examples
- `schemas/`: exported JSON Schemas
- `tests/`: unit, integration, and end-to-end tests
- `docs/`: design and contributor documentation
- `scripts/`: development and release utilities

## Project status

The public APIs beyond the shared T0 conventions are not stable yet. See
[`docs/development-conventions.md`](docs/development-conventions.md) for the
rules that are already frozen.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
