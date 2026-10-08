# AgentQAKit

AgentQAKit is a local-first testing toolkit for AI agents. It is designed to run
repeatable agent tests, collect execution traces, inject controlled failures,
evaluate results, reproduce failures, and enforce quality gates in CI.

The project is currently being built from the ground up. This repository state
contains the T0 foundation: the Python package, command-line entry point,
development conventions, test layout, and CI checks.

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
