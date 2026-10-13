# Specification and artifact contracts

AgentQAKit specifications are strict, versioned YAML documents. `AgentTest`,
`Target`, `Mutation`, and `QualityGate` remain at `agentqa.dev/v1alpha1`.
`AgentManifest` uses `agentqa.dev/v1alpha2`; the loader explicitly migrates the
identity-only v1alpha1 manifest without inventing purpose, capability, policy,
or risk claims.

Run validation before executing a test suite:

```bash
agentqa test validate examples/specs/
```

Validation rejects unknown fields, duplicate YAML keys, unknown evaluator and
target types, invalid cross-field combinations, and unsupported API versions.
Diagnostics identify the source file, YAML document, field path, line, and
column without echoing rejected values.

## AgentManifest v1alpha2

The manifest now has two responsibilities: identify the exact agent build and
provide a bounded semantic contract for AI-assisted test generation. Optional
semantic sections describe `purpose`, public `interfaces`, `capabilities`, tool
side effects and risk, `policies`, `humanInteraction`, quality objectives,
budgets, and non-authoritative `testHints`. `provenance` distinguishes declared,
discovered, inferred, and verified content.

Tools with side effects must declare both `risk` and `idempotent`. A tool with
`requiresConfirmation: true` also requires `humanInteraction.allowed: true`.
Capability names, tool names, and invariant IDs must be unique. All embedded
input/output schemas are checked as JSON Schema Draft 2020-12.

Pass a manifest into a run with:

```bash
agentqa test run tests/ \
  --target targets/agent.yaml \
  --manifest agent-manifest.yaml
```

The run stores `resolved-manifest.json`, containing the normalized declaration
and a bounded runtime snapshot. When the manifest declares a Git commit, the
local checkout is compared and marked `VERIFIED`, `STALE_MANIFEST`, or
`UNVERIFIED`. Drift is reported; declared data is never silently overwritten.

## Templates

Only value lookups are supported:

```yaml
message: "{{ input.message }}"
authorization: "Bearer {{ env.AGENT_API_KEY }}"
```

`input` supports dotted mapping paths. `env` accepts one environment variable
name. Expressions, filters, function calls, attribute access, and arbitrary
Python execution are not supported. Missing variables fail closed. Environment
values have a parallel redacted representation for persistence.

## Stable hashes

Validated specs are serialized to canonical JSON with sorted keys and fixed
separators before SHA-256 hashing. Comments, whitespace, and YAML mapping order
therefore do not affect a spec hash.

## Artifacts

`ArtifactWriter` creates `artifacts/runs/<run_id>/` with trace, cassette, and
failure subdirectories. JSON files include `schema_version` and have adjacent
`.sha256` files. JSONL result records contain a checksum of their payload.

Files are written to a temporary file in the same directory, flushed, and
atomically replaced. `summary.json` finalizes a run; later writes through that
writer are rejected. Credential-like mapping keys are redacted before data is
serialized.

The generated contracts in [`schemas/`](../schemas/) can be refreshed with:

```bash
python scripts/export_schemas.py
```

## Evaluation expectations

`expect` is declarative data; it never executes user-provided Python. Available
sections are:

- `output`: Draft 2020-12 JSON Schema, exact field paths, required and forbidden
  regex patterns, refusal intent, non-empty output, and allowed HTTP statuses;
- `tools`: required or forbidden names, total/per-tool call limits, argument
  schemas, and pairwise `before`/`after` constraints;
- `trajectory`: maximum `ERROR` spans and required span attributes;
- `state.all`: restricted `equals`, `notEquals`, `exists`, `in`, and `before`
  invariants;
- `budgets`: latency, token, step, LLM/tool call, cost, and retry limits.

For example:

```yaml
expect:
  output:
    jsonSchema:
      type: object
      required: [answer]
    exactFields:
      $.status: ok
    allowedHttpStatuses: [200]
  tools:
    required: [course.lookup]
    argumentSchemas:
      course.lookup:
        type: object
        required: [course_id]
    order:
      - before: course.lookup
        after: answer.render
  trajectory:
    maxErrorSpans: 0
  state:
    all:
      - field: data_disclosed
        op: equals
        value: false
  budgets:
    maxLatencyMs: 8000
    maxLlmCalls: 3
    maxToolCalls: 4
    maxRetries: 1
```

Field paths support only names and integer indexes, such as `$.answer.text` or
`items[0].id`. Evaluator behavior and observability rules are documented in
[`evaluators.md`](evaluators.md).

Target-specific runtime behavior and network/process security defaults are
documented in [`targets.md`](targets.md). Mutation operations, matching fields,
runtime evidence, and scoring are documented in [`mutations.md`](mutations.md).
