# AgentQAKit

AgentQAKit is a local-first testing toolkit for AI agents. It is designed to run
repeatable agent tests, collect execution traces, inject controlled failures,
evaluate results, reproduce failures, and enforce quality gates in CI.

The project is currently being built from the ground up. The repository now
contains the package foundation, strict specifications, concurrent budget-aware runner,
HTTP, isolated Python, and optional LangGraph target adapters, plus bounded
OpenTelemetry trace collection, OTLP/HTTP instrumentation, deterministic tool,
retriever, and HTTP mutations, and recomputable Mutation Score reports.
Isolated Python agents can opt into those same trace, mutation, and replay
capabilities with `@agentqa.agent`, `@agentqa.tool`, and
`@agentqa.retriever` decorators.
It also includes deterministic output, tool, trajectory, state, privacy, and
budget evaluators plus an opt-in, provider-neutral LLM judge boundary, canonical
dependency replay, and verified failure reproduction bundles.
It also compares verified runs and enforces declarative quality gates with
CI-friendly exit codes and checksummed reports.
Manifest-grounded AI generation can now produce a reviewable capability-risk
test plan, strict candidate `AgentTest` specifications, and a checksummed
coverage report without granting AgentQAKit ambient source-tree access.
The same bounded evidence layer grounds semantic Judge decisions and produces
separate, evidence-citing AI failure-analysis reports for failed trials.
An optional local MCP server now exposes those same application services to
Coding Agents through bounded, checksummed resources and approval-gated tools,
without granting arbitrary filesystem or shell access.
The complete standalone, MCP collaboration, and controlled three-arm T16
experiment is documented in [`docs/t16-end-to-end.md`](docs/t16-end-to-end.md).
Completed runs can now be indexed in a checksummed file store or transactional
SQLite database, queried through a local FastAPI service, and inspected in the
React evidence console.

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
# Optional local Coding Agent collaboration over MCP:
python -m pip install -e ".[mcp]"
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
  --target examples/specs/http-target.yaml \
  --manifest examples/specs/agent-manifest.yaml
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

Compare a baseline and candidate, then enforce a quality gate:

```bash
agentqa compare artifacts/baseline/runs/<id> artifacts/candidate/runs/<id> \
  --output artifacts/comparison.json
agentqa gate evaluate quality-gate.yaml \
  --baseline artifacts/baseline/runs/<id> \
  --result artifacts/candidate/runs/<id>
```

Build and launch the local evidence console:

```bash
cd apps/web
npm ci
npm run build
cd ../..
agentqa serve
```

The server binds to `127.0.0.1:8000`, stores indexes under `.agentqa/`, and
accepts only workspace-relative specification paths by default. The API is also
available without the UI at `uvicorn apps.api.main:app`.

Enable semantic evaluation with DeepSeek by copying `.env.example` to the
Git-ignored `.env.local`, setting its permissions to `600`, and running:

```bash
agentqa test run examples/specs/ci-test.yaml \
  --target examples/specs/ci-candidate-target.yaml \
  --judge-provider deepseek \
  --no-trace-receiver
```

The API key is process-only secret input. Structured LLM evidence records model
identity, configuration and evidence hashes, bounded request IDs, verdict,
rationale, latency, attempts, and token usage, but never the key or raw prompt.

Generate a test plan and validated candidate tests with the same DeepSeek
configuration:

```bash
agentqa generate plan \
  --manifest examples/specs/agent-manifest.yaml \
  --goal examples/specs/generation-goal.yaml \
  --output artifacts/generation/plan.json
agentqa generate tests \
  --manifest examples/specs/agent-manifest.yaml \
  --plan artifacts/generation/plan.json \
  --output tests/generated
agentqa test validate tests/generated
```

High- and critical-risk generated tests are marked `REVIEW_REQUIRED` and are
rejected by the run service until a reviewer changes that label to `APPROVED`.
The two-phase workflow, artifacts, CodeEvidence boundary, and safety checks are
documented in [`docs/generation.md`](docs/generation.md).

Analyze a failed trial without changing its original verdict or trace:

```bash
agentqa analyze failure \
  --run artifacts/runs/<run-id> \
  --trial-id <failed-trial-id> \
  --output artifacts/analysis/<trial-id>.json
```

The report cites stable evidence IDs for facts and root-cause candidates and
records missing evidence explicitly. See
[`docs/failure-analysis.md`](docs/failure-analysis.md).

Expose the project to a local MCP host through the official stdio transport:

```bash
agentqa mcp serve \
  --workspace /absolute/path/to/agent-project \
  --manifest agent-manifest.yaml
```

The server offers manifest and schema resources, bounded run/failure evidence,
and tools for generation, validation, execution, mutation, analysis,
comparison, and gates. Setup, host configuration, the `CodeEvidence` contract,
and side-effect approval rules are documented in
[`docs/mcp.md`](docs/mcp.md).

AgentQAKit accepts strict, versioned YAML. Unknown fields and unsupported
evaluator or target types are rejected instead of being ignored. See
[`docs/specifications.md`](docs/specifications.md) for the supported format.
The Manifest v2 field reference and migration checklist are in
[`docs/manifest-v2.md`](docs/manifest-v2.md).

The runner provides an explicit lifecycle, bounded trial concurrency, global
Token/cost budgets, per-trial isolation, timeouts, cancellation,
partial-failure handling, and stability aggregation.
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
Run alignment, removed-test semantics, metric observability, gate rules, and CI
integration are documented in
[`docs/comparison-and-gates.md`](docs/comparison-and-gates.md).
The evidence boundary and immutable AI failure-analysis workflow are documented
in [`docs/failure-analysis.md`](docs/failure-analysis.md).
Local MCP resources, tools, workspace confinement, and explicit side-effect
approval are documented in [`docs/mcp.md`](docs/mcp.md).
Result-store guarantees, SQLite layout, HTTP endpoints, server security, and
Web development are documented in
[`docs/storage-api-web.md`](docs/storage-api-web.md).

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
