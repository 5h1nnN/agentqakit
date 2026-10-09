# Specification and artifact contracts

AgentQAKit specifications are strict, versioned YAML documents. The current API
version is `agentqa.dev/v1alpha1`. Supported kinds are `AgentTest`, `Target`,
`Mutation`, and `AgentManifest`.

Run validation before executing a test suite:

```bash
agentqa test validate examples/specs/
```

Validation rejects unknown fields, duplicate YAML keys, unknown evaluator and
target types, invalid cross-field combinations, and unsupported API versions.
Diagnostics identify the source file, YAML document, field path, line, and
column without echoing rejected values.

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

Target-specific runtime behavior and network/process security defaults are
documented in [`targets.md`](targets.md).
