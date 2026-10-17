# Web, MCP and Coding Agent handoff

## One artifact contract

Web and MCP persist the same strict domain models for generation plans/reports,
runs, analyses, comparisons and gates. Each authoritative file has a neighboring
`.sha256` file. The Artifact Catalog records the payload checksum, schema version,
source and availability; it does not rewrite the source artifact.

Use **Discover artifacts** to import `.agentqa/mcp`, CLI runs, benchmark results
or reference results. Imported artifacts can then be selected by later Web
steps. An MCP resource wraps an authoritative artifact for transport, but exposes
the artifact's original checksum and exact schema-shaped value under `artifact`.

## Bounded repair handoff

The Web workbench creates:

```text
agentqa://workflows/<workflow-id>/handoffs/<handoff-id>
```

The resource contains only evidence URIs, immutable baseline hashes, missing
evidence, suggested regression-test titles, and the redacted developer task. It
does not contain source excerpts, prompts, credentials or arbitrary logs.

A Coding Agent should:

1. read the Handoff Bundle through MCP;
2. read its failure and analysis resource URIs;
3. inspect the repository using its own explicitly granted workspace access;
4. implement and test a bounded repair;
5. return control to the user.

AgentQAKit then reruns project preflight and records the new source/runtime
identity through ordinary run evidence. The Coding Agent does not write arbitrary
success claims into AgentQAKit's database.

AgentQAKit's MCP surface deliberately has no generic filesystem write, shell,
Git, patch, or database tool. This keeps QA evidence independent from the agent
that changes the product.

## Data boundary

- Manifest sections and explicitly selected CodeEvidence may be sent to an AI
  provider only after the Web preview is shown.
- API keys stay in process environment and are never artifact fields.
- Handoff and WorkflowReport are bounded exchange artifacts and may contain
  hashes and URIs, but not raw CodeEvidence or prompts.
- Targets declaring external or irreversible effects require exact, expiring
  approval. Use synthetic/sandbox accounts for demonstrations.
- The service is designed for one trusted local user. It has no multi-tenant
  authentication boundary and must not be exposed on a public interface.

Run `make leak-scan` to check persisted exchange artifacts for common key/header
patterns and over-broad raw-code/prompt fields.
