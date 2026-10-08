# Contributing to AgentQAKit

Thank you for contributing. AgentQAKit is in an early development phase, so
small, testable changes that preserve the documented interfaces are preferred.

## Set up the repository

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before opening a pull request

Run the same checks as CI:

```bash
make lint
make test
make e2e
```

Every behavior change should include a unit test and, where applicable, an
error-path test. Update public documentation when changing an API or
configuration field. Never commit API keys, credentials, private user data, or
generated runtime artifacts.

## Code conventions

The frozen conventions for exceptions, timestamps, identifiers, serialization,
timeouts, artifacts, and error handling are documented in
[`docs/development-conventions.md`](docs/development-conventions.md).

Use focused commit messages such as:

```text
feat(runner): add sequential test execution
fix(trace): preserve parent context across async calls
test(replay): add missing-cassette fail-closed case
```

By submitting a contribution, you agree that it is licensed under the Apache
License, Version 2.0.
