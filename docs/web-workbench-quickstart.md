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

Open `http://127.0.0.1:8000`, choose **Workbench**, and keep the terminal open.
DeepSeek is optional for browsing, Campaigns, comparisons and gates. To use AI
generation or analysis, configure `.env.local` as described in the README before
starting the server. Never paste a key into the browser.

## 2. Create the Campus Ops project

Create a project with ID `campus-ops`, directory `.`, and these defaults:

| Field | Workspace-relative path |
|---|---|
| Manifest | `examples/showcase/campus_ops_agent/agentqa/manifest-v2.yaml` |
| Tests | `examples/showcase/campus_ops_agent/agentqa/golden-tests/campus-ops.yaml` |
| Mutations | `examples/showcase/campus_ops_agent/agentqa/mutations.yaml` |
| Defect target | `examples/showcase/campus_ops_agent/agentqa/targets/bug-authorization.yaml` |
| Fixed target | `examples/showcase/campus_ops_agent/agentqa/targets/fixed.yaml` |
| Gate | `examples/showcase/campus_ops_agent/agentqa/regression-gate.yaml` |
| Generation goal | `examples/showcase/campus_ops_agent/agentqa/generation-goal.yaml` |

Select **Run preflight**. Continue only when it says `EXECUTABLE`; review any
side-effect or provider warning rather than treating it as a pass.

## 3. Select evidence and create a workflow

Use **Code evidence** to preview and explicitly add the seven excerpts listed in
`examples/showcase/campus_ops_agent/agentqa/code-evidence/t16-code-evidence.json`.
Previewing does not authorize an AI call; **Add to CodeEvidence** does. Then
create a READY workflow and select it.

In **Generation & review studio**, choose Manifest + code, check the seven
excerpts, generate the plan, confirm it, and generate candidates. Approve low or
medium risk cases in a batch. Review high and critical risk cases individually.

For a no-LLM rehearsal, skip generation and use the checked-in golden tests.

## 4. Run, reproduce and diagnose

In **Campaign builder**:

1. select the fixed and one defect target;
2. enable Mutation when required;
3. choose `record`, freeze the config, inspect the risk summary, generate the
   one-time phrase, type it, and start;
4. for replay, create a second Campaign with `replay` and the recorded cassette;
5. if a Campaign stops at `INTERRUPTED` or `INCOMPLETE`, use **Retry from safe
   checkpoint**; completed subruns are not silently repeated.

In **Failure Inbox**, select a failure, preview the exact redacted evidence, and
explicitly start AI Analysis. Facts and hypotheses remain separate and every AI
claim cites evidence IDs. Generate the Handoff Bundle and copy its MCP URI and
human-readable task for the Coding Agent.

## 5. Simulate repair and release

After an external repair, rerun preflight. AgentQAKit compares the new manifest,
source and runtime hashes; it does not accept arbitrary log writeback as proof.
Resume the workflow, select the fixed target, and rerun. Then:

1. compare defect/baseline with fixed/candidate;
2. run the regression gate;
3. run the mutation gate;
4. generate and download `WorkflowReport`.

`UNOBSERVABLE` is intentional: missing token, cost, trace or replay evidence is
never converted to zero. Use **Discover artifacts** to import prior CLI, MCP,
benchmark or reference results into the same catalog.

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
