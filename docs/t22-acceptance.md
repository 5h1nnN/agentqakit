# T22 product acceptance

## Automated acceptance

```bash
python -m pip install -e ".[dev,mcp]"
cd apps/web && npm ci && npx playwright install chromium && cd ../..
make acceptance
```

The target includes Python lint/type checks, the full Python suite, Vitest,
production Web build, schema export, artifact leak scanning and Playwright's
no-network Campus Ops Golden Path. `make e2e` and MCP integration tests remain
available as focused diagnostics.

The Playwright scenario intercepts only `/api/**`, uses deterministic Mock
Provider fixtures, and exercises the actual production React controls for
preflight, seven CodeEvidence selections, generation/review state, completed
fixed+defect mutation/record-replay evidence, failure evidence authorization,
analysis reuse, Handoff, repair identity, comparison, gate and WorkflowReport.

## Real DeepSeek opt-in benchmark

Real-provider validation is manual because it incurs cost and depends on an
external service. Configure `.env.local` with mode `600`, start `agentqa serve`,
and follow `web-workbench-quickstart.md`. An explicitly authorized automated
run can use `AGENTQA_REAL_DEEPSEEK=1`, `PLAYWRIGHT_EXTERNAL_SERVER=1`, and
`PLAYWRIGHT_BASE_URL=<local server>` with `npm run test:e2e -- --grep "real DeepSeek"`.
Save the final WorkflowReport and all
sidecars under:

```text
benchmark/reference-results/t22-deepseek-web-<date>/
```

Record provider/model, source/runtime hashes, evidence hashes, generated case
count, Campaign run IDs, comparison ID, both gate results and WorkflowReport
checksum in `acceptance.json`. Do not save the key, raw prompts or unselected
source. Run:

```bash
python scripts/scan_artifact_leaks.py benchmark/reference-results/t22-deepseek-web
```

The reviewer verifies all adjacent checksums and confirms the run was performed
through Web controls rather than helper scripts or SQLite editing.

### Accepted reference run

The real Web run completed on 2026-10-16 (Asia/Shanghai) is stored at
`benchmark/reference-results/t22-deepseek-web-20261015/`; the directory name
uses the UTC start date recorded by the run. It used `deepseek-flash` with
structured-output thinking disabled, generated and approved six candidates,
ran 33 defect/target/fixed mutation trials, persisted one failure analysis and
MCP-addressable handoff, replayed one failed trial with one cassette hit and no
live fallback, and ran a dedicated fixed-target mutation campaign. The final
regression gate passed with exit code `0`; the final mutation gate passed at the
configured `0.8` mutation-score threshold with exit code `0`. In total the
reference report contains three Campaigns, five runs and 45 trials. The audit
history retains an earlier failed-safe mutation-gate evaluation whose score was
unobservable, followed by the evidence-producing Campaign and passing reevaluation.
The report checksum is recorded in both `workflow-report.json.sha256` and
`acceptance.json`.

## Release checklist

- Mock browser Golden Path passes twice from a clean install.
- Web and MCP resolve the same Analysis/Handoff payload and artifact checksum.
- Timeout, invalid JSON, restart, partial Campaign and external-repair recovery
  are visible and documented.
- Quickstart succeeds for a reviewer unfamiliar with the repository.
- Real DeepSeek Web evidence exists when network/cost authorization is granted.
- `git status` contains only intentional source, lockfile, schema and reference
  artifact changes; no test traces, databases, `.env.local`, keys or caches.
