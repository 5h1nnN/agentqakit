# Local MCP Server

AgentQAKit's optional MCP server lets a Coding Agent use the same generation,
runner, mutation, comparison, gate, and failure-analysis application services as
the CLI. The first release is deliberately local-only: it uses the official
Python MCP SDK over `stdio` and fixes one workspace boundary when the server
starts.

## Install and start

Install the optional dependency:

```bash
python -m pip install -e ".[mcp]"
```

Start the server with an absolute workspace path:

```bash
agentqa mcp serve \
  --workspace /absolute/path/to/agent-project \
  --manifest agent-manifest.yaml
```

`--manifest` is relative to the fixed workspace. If the workspace contains a
`.env.local`, the server uses the existing DeepSeek configuration for AI test
generation, Judge evaluation, and failure analysis. A different private file
can be selected with `--llm-env-file`. Non-AI validation, execution,
comparison, and gate tools work without an LLM configuration.

For an MCP host that accepts JSON server definitions, the equivalent entry is:

```json
{
  "mcpServers": {
    "agentqakit": {
      "command": "agentqa",
      "args": [
        "mcp",
        "serve",
        "--workspace",
        "/absolute/path/to/agent-project",
        "--manifest",
        "agent-manifest.yaml"
      ]
    }
  }
}
```

The MCP protocol owns standard output. Operational logs go only to standard
error and contain the tool name, request ID, optional run ID, and outcome, not
prompts, evidence contents, or credentials.

## Resources

The server publishes these read-only resources:

| URI | Content |
| --- | --- |
| `agentqa://project/manifest` | Validated and redacted project Manifest |
| `agentqa://schemas/agent-test` | Current `AgentTest` JSON Schema |
| `agentqa://schemas/agent-manifest` | Current `AgentManifest` JSON Schema |
| `agentqa://runs/{run_id}/summary` | Verified bounded run summary |
| `agentqa://runs/{run_id}/trace-summary` | Counts, statuses, tool names, and error counts; never full events |
| `agentqa://runs/{run_id}/failures/{failure_id}` | Verified failure-bundle summary |
| `agentqa://comparisons/{comparison_id}` | Verified comparison report |

Every resource is redacted, size-bounded, and wrapped with
`schema_version`, `uri`, `checksum`, and `data`. Missing or invalid dynamic
resources return an error object inside the same checksummed envelope, so an
MCP client does not need to parse server tracebacks.

## Tools

| Tool | Purpose |
| --- | --- |
| `validate_manifest` | Validate a Manifest and return bounded diagnostics |
| `generate_tests` | Generate a plan and validated test drafts under `.agentqa/mcp/drafts/` |
| `validate_tests` | Validate one file or directory of test specifications |
| `run_tests` | Execute tests through the shared runner |
| `run_mutations` | Execute tests with a declared mutation suite |
| `get_run_result` | Return a small verified result plus follow-up resource URIs |
| `analyze_failure` | Write a separate evidence-citing AI analysis report |
| `compare_runs` | Compare two verified runs and write a checksummed report |
| `evaluate_gate` | Evaluate an existing Gate specification and write a report |

Tool parameters use `snake_case`, such as `tests_path`, `target_path`, and
`allow_side_effects`. Paths are always relative to the configured workspace.
Run tools accept only specification paths; they do not accept arbitrary shell
commands or entrypoints.

## Coding Agent evidence

`generate_tests` and `analyze_failure` optionally accept `code_evidence`. The
Coding Agent reads the repository, selects only relevant excerpts, and sends
the excerpt itself. AgentQAKit never opens the supplied `path` automatically.

```json
{
  "code_evidence": [
    {
      "path": "src/agent/tools.py",
      "startLine": 40,
      "endLine": 52,
      "contentHash": "sha256:<64 lowercase hex characters>",
      "reason": "Defines the non-idempotent payment tool",
      "content": "def charge_card(...): ..."
    }
  ]
}
```

`contentHash` is the SHA-256 digest of the exact UTF-8 `content`, including the
`sha256:` prefix. A path containing `..`, an absolute path, a mismatched hash,
an oversized excerpt, too many excerpts, or duplicate evidence hashes is
rejected before any LLM call. Existing secret redaction is applied before
evidence enters an LLM request.

## Side-effect approval

`run_tests` and `run_mutations` default to `allow_side_effects=false`. If the
Manifest declares any tool whose `sideEffect` is not `none`, the call is
rejected unless the caller explicitly sets `allow_side_effects=true`. An
`external-write` or `irreversible` tool also requires a non-empty
`approval_id` that identifies the human approval handled by the MCP host.

This approval is an execution guard, not a capability grant. The runner still
loads exactly one validated `TargetSpec`, verifies every test's `targetRef`,
uses the fixed artifact directory, and preserves all existing budgets,
timeouts, isolation, trace, mutation, and evaluator behavior.

The server intentionally provides no tool for arbitrary file reads or writes,
shell execution, source modification, Git operations, artifact deletion,
upload, or remote MCP transport.

## Suggested collaboration flow

1. Read `agentqa://project/manifest` and the relevant JSON Schema.
2. Validate the Manifest and existing tests.
3. Optionally select bounded code excerpts and call `generate_tests`.
4. Review generated high- or critical-risk drafts before approving them.
5. Call `run_tests` or `run_mutations` with workspace-relative spec paths.
6. Follow the returned summary and trace-summary resource URIs.
7. For a failure, read its failure resource and optionally call
   `analyze_failure` with additional selected code evidence.
8. Compare the run with a baseline and evaluate the project's quality gate.

## Troubleshooting

- `MCP_LLM_NOT_CONFIGURED`: configure the existing DeepSeek environment before
  using generation or analysis; non-AI tools remain available.
- `MCP_PATH_ERROR`: use a workspace-relative POSIX path without `..` or a
  symlink that escapes the workspace.
- `MCP_SIDE_EFFECTS_NOT_ALLOWED`: review the Manifest and explicitly authorize
  the run only if the declared effects are acceptable.
- `MCP_APPROVAL_REQUIRED`: provide the MCP host's human approval identifier for
  external or irreversible tools.
- `RUN_SERVICE_TARGET_MISMATCH`: the test's `targetRef` does not name the
  selected `TargetSpec`.
- `MCP_RESOURCE_NOT_FOUND`: the requested run, failure, comparison, or report
  is not present under the fixed workspace.

