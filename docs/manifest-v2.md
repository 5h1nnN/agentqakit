# Manifest v2 field and migration guide

`AgentManifest` `agentqa.dev/v1alpha2` is the semantic contract between a tested Agent,
AgentQAKit, and an optional Coding Agent. Source access does not replace it: source explains
implementation, while the Manifest declares reviewed intent, risk, permissions, side effects,
budgets, and stable names that tests may reference.

## Field guide

| Field | Required | Purpose |
| --- | --- | --- |
| `metadata` | yes | Stable Agent name, version, and labels. |
| `runtime` | yes | Language, framework, entrypoint, and dependency identity. |
| `source` | no | Repository, commit, dirty state, and source hash used for provenance. |
| `model` / `prompts` | no | Model settings and prompt hashes without private raw prompts. |
| `purpose` | no | Intended use and explicit out-of-scope behavior. |
| `interfaces` | no | JSON-compatible input and output schemas. |
| `capabilities` | no | User-visible operations used by the generation coverage matrix. |
| `tools` | no | Tool schema, side effect, risk, idempotency, and confirmation boundary. |
| `policies` | no | Invariants, forbidden behaviors, and permission requirements. |
| `humanInteraction` | no | Whether intervention exists, when required, and its budget. |
| `quality` | no | Objectives and latency, step, call, retry, token, or cost budgets. |
| `knowledge` | no | Whether retrieval exists and which index snapshot is under test. |
| `testHints` | no | Reviewed high-value scenarios; hints never override policy. |
| `provenance` | no | Declared/discovered/inferred confidence and review status. |

Tool `sideEffect` is the execution-safety field. `none` is read-only;
`local-write`, `external-write`, and `irreversible` require increasingly explicit review. MCP
blocks any declared effect by default, and external or irreversible tools additionally require a
human approval identifier.

## Migration from v1alpha1

1. Keep identity and runtime fields, then change `apiVersion` to `agentqa.dev/v1alpha2`.
2. Add `purpose.summary`; write `outOfScope` for requests the Agent must reject.
3. Promote each public behavior to a stable `capabilities[].name`.
4. Describe each observable tool. Set `sideEffect` conservatively, then add `risk`,
   `idempotent`, `requiresConfirmation`, and its input schema where known.
5. Convert prose requirements into named `policies.invariants`. Names become stable generation
   and coverage identifiers, so changing one is a contract migration.
6. Add HITL rules and quality budgets. Do not invent values merely to fill optional fields.
7. Pin source/model/prompt/index hashes when reproducibility matters.
8. Validate with `agentqa test validate manifest-v2.yaml`, inspect the normalized hash, and have a
   human review inferred or code-discovered claims.

The runtime writes the resolved Manifest and its hash into each run. A generation plan refuses a
different Manifest hash, preventing silent execution against a changed contract. See the complete
`examples/showcase/campus_ops_agent/agentqa/manifest-v2.yaml` and its deliberately simplified
comparison Manifest.

