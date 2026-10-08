# Development conventions

These rules are frozen for the P0 implementation. Changes require an explicit
interface decision and corresponding documentation and test updates.

## Errors

- `agentqa.errors.AgentQAError` is the base class for expected AgentQAKit
  failures.
- P0 code must not silently discard exceptions. It must either handle the
  failure completely or raise/translate it with enough context to diagnose.

## Time and identifiers

- Persisted timestamps use timezone-aware UTC and ISO 8601.
- The canonical serialized UTC suffix is `Z`; for example,
  `2026-10-09T08:30:00.000000Z`.
- Persisted entity identifiers use UUIDs. `agentqa.conventions.new_id()` returns
  their canonical string representation.

## Serialization

- JSON field names use `snake_case`.
- User-facing YAML fields use `camelCase`.
- Pydantic models map YAML aliases to the internal `snake_case` field names.
- Unknown configuration fields are rejected rather than ignored.

## External calls

- Every network request, subprocess, model call, and tool call has an explicit
  timeout. There is no implicit infinite wait.
- Timeout values must be positive and use a unit in their field name or schema
  description.

## Artifacts and secrets

- Runtime artifacts never store raw credentials, authorization headers,
  cookies, API keys, or unrelated environment variables.
- Sensitive values are redacted before serialization and logging.
- Checked-in examples use synthetic data and placeholder credentials only.
