# T16 end-to-end workflows and controlled experiment

T16 uses `examples/showcase/campus_ops_agent`: a fixed LangGraph Agent, five one-defect variants,
21 human golden tests, 11 fault mutations, reviewed Manifest v2, bounded CodeEvidence, and defect
truth kept outside model prompts.

## Independent golden path

```bash
python -m pip install -e ".[dev,langgraph,mcp]"
python -m pip install --no-deps --no-build-isolation -e examples/showcase/campus_ops_agent
chmod 600 .env.local
python scripts/run_t16.py \
  --output benchmark/results/t16-deepseek-20261014 \
  --approve-high-risk \
  --max-generated-cases 6 \
  --llm-max-tokens 16384 \
  --llm-timeout-seconds 180
```

By default, only AI-generated low/medium-risk tests execute. High/critical tests remain
`REVIEW_REQUIRED` and are recorded as rejected. `--approve-high-risk` explicitly changes their
copied review label to `APPROVED`; it does not change the immutable generation report.

The script performs Manifest-grounded two-phase generation, reviewed execution against fixed and
seeded-defect targets, the human golden suite in record mode, Mutation Score, one
deterministic-plus-Judge case, evidence-grounded failure analysis, a verified failure/replay
bundle, comparison, and separate regression and mutation gates. Standalone analysis receives no
CodeEvidence, proving the toolkit can close the loop without a Coding Agent.

Outputs include `environment.json`, `summary.json`, `metrics.csv`, plans, generation reports,
review actions, run artifacts, cassettes, failure ZIPs, analysis, comparisons, gates, and SHA-256
sidecars. The script refuses a non-empty output directory by default. `--resume` verifies and
reuses completed stage checkpoints; it does not silently trust unfinished output.

## MCP collaboration golden path

```bash
python scripts/run_t16_mcp.py \
  --output benchmark/results/t16-mcp-deepseek-20261014 \
  --approval-id t16-human-review-20261014 \
  --max-generated-cases 4 \
  --llm-max-tokens 16384 \
  --llm-timeout-seconds 180
```

It copies the showcase into an isolated experiment workspace, then uses official MCP resources
and tools to read the Manifest, submit selected CodeEvidence, generate and validate tests, run a
seeded defect, read failure evidence, analyze it, rerun the fixed implementation, execute
mutations, compare runs, and evaluate both gates. The checked transcript contains tool outputs
and resource hashes, never the API key.

The MCP golden path defaults to four generated cases because that bounded transaction completed
reliably with strict JSON validation in the verified DeepSeek run. It still executes the complete
21-case human golden suite and all 11 mutation scenarios; the generated-case budget limits only
the AI-authored draft suite. Larger generated suites should be split into multiple transactions.

The script represents repair with the pre-reviewed fixed variant so the experiment is
deterministic. In an interactive session, a Coding Agent reads the same analysis, edits the tested
project using its own workspace tools, then reruns. AgentQAKit's MCP server deliberately has no
source-editing or shell tool. See [mcp.md](mcp.md) for stdio host configuration and troubleshooting.

## Three-arm comparison

All arms use the same provider settings, goal, six-case budget, seed, target, mutations, and
evaluation semantics. The independent variable is:

1. `minimal-manifest`: two capabilities and two tools, without policies or schemas;
2. `manifest-v2`: the reviewed semantic contract;
3. `manifest-v2-code-evidence`: the same contract plus seven hashed, caller-selected excerpts.

For each arm, `summary.json` records valid tests, invalid-output repair batches, duplicate and
hallucinated-tool cases that survived validation, capability/policy/risky-tool coverage, five
seeded-defect outcomes, generated Mutation Score, review actions, tokens, estimated cost per valid
test, Manifest hash, context hashes, generation ID, and run IDs. Invalid raw model payloads are not
persisted; safe call evidence records the repair attempt and status.

The six-case budget keeps one structured DeepSeek request within the bounded output window. A
larger experiment should partition the goal into multiple generation batches instead of weakening
validation or accepting truncated JSON.

Defect detection is conservative: a test must pass on the fixed Agent and fail on the relevant
defect variant. Mutation kills remain separate from ordinary pass rate. `NOT_REACHED` mutations
are excluded from the Mutation Score denominator.

## DeepSeek configuration, cost, and data boundary

AgentQAKit sends the normalized Manifest, approved goal, selected CodeEvidence, schema, and
bounded evaluation/failure evidence to DeepSeek. It does not send the repository, environment,
API key, arbitrary files, or raw secret fields. Prompts and responses are not stored; hashes,
request IDs, model parameters, token usage, latency, status, and structured outputs are stored.

Cost is estimated from provider-returned cache-hit, cache-miss, and output tokens. T16 defaults
pin the DeepSeek Flash off-peak USD-per-million snapshot observed on 2026-10-14 (`0.003`, `0.15`,
`0.60`). Pricing is time- and model-dependent, so values and the official pricing URL are stored
in `environment.json`; override all three CLI prices under another model or price window. Provider
usage is the token source of truth.

## Web inspection

Start `agentqa serve`, open the local console, and select **AI evidence**. The API scans only known
workspace-local artifact layouts and accepts reports with valid SHA-256 sidecars and strict
schemas. It shows generation Manifest/plan/context hashes, model calls, token usage, coverage gaps,
case risk and approval state; AnalysisReport evidence hashes, ranked causes, confidence, citations,
and gaps; and the existing run views for Judge, traces, mutations, findings, comparisons, and gates.
The endpoint is read-only and returns bounded summaries.

## Acceptance verification

```bash
make lint
make test
cd apps/web && npm test -- --run && npm run build && cd ../..
pytest -q tests/integration/test_mcp_server.py
python scripts/export_schemas.py --check
```

Keep `benchmark/results/` as raw local output. After checking hashes, secrets, and completeness,
copy the selected immutable run to `benchmark/reference-results/` with its README/data card.

