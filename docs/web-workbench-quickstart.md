# Web Workbench five-minute quickstart

This guide runs AgentQAKit's local Web workflow without helper scripts or manual
database edits. The workbench is local-single-user software: bind it only to
`127.0.0.1` and use workspace-relative paths.

## 1. Start the workbench

```bash
python -m pip install -e ".[dev,mcp]"
cd apps/web && npm ci && npm run build && cd ../..
agentqa serve
```

Open `http://127.0.0.1:8000`, choose **Workspace**, and keep the terminal open.
DeepSeek is optional for browsing, Campaigns, comparisons and gates. To use AI
generation or analysis, configure `.env.local` as described in the README before
starting the server. Never paste a key into the browser.

## 2. Connect the Campus Ops repository

Choose **New project** and enter only:

| Field | Value |
|---|---|
| Project name | `Campus Ops` |
| Repository directory | `examples/showcase/campus_ops_agent` |

Project ID and description are optional. **Create and scan** finds the AgentQA
manifest, default target, golden tests, mutations, gate, generation goal, and
known defect/fixed variants by their validated specification kinds. It also
creates the first evaluation. In **Overview**, choose **Scan project** and
continue when Readiness says **Ready**. Open Readiness details only when a check
needs attention.

## 3. Build repository context and generate tests

Open **Test design** and choose **Build AI context**. AgentQAKit scans safe text
source files, ignores generated, secret, data and test paths, ranks files using
agent-structure signals, bounds their size, and redacts likely secrets. Review
the selected file list before using it. This replaces manual line-by-line code
excerpt collection; the low-level CodeEvidence API remains available for SDK and
MCP integrations.

In **Generate and review tests**, choose **Repository context**, describe the
goal, and use the purple AI button to generate a plan. Confirm the plan and
generate candidates. Approve low or medium risk cases in a batch. Review high
and critical risk cases individually.

For a no-LLM rehearsal, skip generation and use the checked-in golden tests.

## 4. Run, reproduce and diagnose

Open **Execute**. In **Campaign builder**:

1. select the fixed and one defect target;
2. enable Mutation when required;
3. choose `record`, freeze the config, inspect the risk summary, generate the
   one-time phrase, type it, and start;
4. for replay, create a second Campaign with `replay` and the recorded cassette;
5. if a Campaign stops at `INTERRUPTED` or `INCOMPLETE`, use **Retry from safe
   checkpoint**; completed subruns are not silently repeated.

Open **Investigate**, select a failure, review the exact redacted evidence, and
use **Analyze failure with AI**. Facts and hypotheses remain separate and every
AI claim cites evidence IDs. Generate the Handoff Bundle and copy its MCP URI
and human-readable task for the Coding Agent.

## 5. Simulate repair and release

After an external repair, rerun preflight. AgentQAKit compares the new manifest,
source and runtime hashes; it does not accept arbitrary log writeback as proof.
Resume the evaluation, select the fixed target, and rerun. Open **Release** and:

1. compare defect/baseline with fixed/candidate;
2. run the regression gate;
3. run the mutation gate;
4. generate and download `WorkflowReport`.

Missing token, cost, trace or replay evidence is never converted to zero. Use
**Refresh history** in Overview to import prior CLI, MCP, benchmark or reference
results into the same catalog. Technical hashes and raw records are available
inside expandable details instead of dominating the primary workflow.

## Recovery map

| Condition | Safe recovery |
|---|---|
| service restart | reopen the workflow; RUNNING work is marked interrupted and can be resumed |
| LLM timeout/provider error | inspect FAILED_SAFE, keep the evidence hash, and retry explicitly |
| invalid model JSON | no candidate is accepted; retry creates a new immutable attempt |
| partial Campaign | retry from its checkpoint; retain completed subrun evidence |
| external repair | rerun preflight, verify new source/runtime identity, resume, then rerun |

Run `make acceptance` before release. The browser test uses a deterministic,
no-network Mock Provider; real DeepSeek is intentionally opt-in.
