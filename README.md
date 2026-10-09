# AgentQAKit

AgentQAKit is a local-first testing toolkit for AI agents. It is designed to run
repeatable agent tests, collect execution traces, inject controlled failures,
evaluate results, reproduce failures, and enforce quality gates in CI.

The project is currently being built from the ground up. The repository now
contains the package foundation, strict specifications, sequential runner, and
HTTP, isolated Python, and optional LangGraph target adapters.

## Requirements

- Python 3.11 or newer
- `pip`

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
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

AgentQAKit accepts strict, versioned YAML. Unknown fields and unsupported
evaluator or target types are rejected instead of being ignored. See
[`docs/specifications.md`](docs/specifications.md) for the supported T1 format.

The sequential runner provides an explicit lifecycle, per-trial isolation,
timeouts, cancellation, partial-failure handling, and stability aggregation.
See [`docs/runner.md`](docs/runner.md). Target construction, security defaults,
subprocess isolation, and LangGraph observation are documented in
[`docs/targets.md`](docs/targets.md).

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
