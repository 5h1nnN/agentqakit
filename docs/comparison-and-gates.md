# Comparison and quality gates

AgentQAKit compares completed, checksummed run directories. Both arguments must
be `artifacts/runs/<run-id>` directories produced by the runner. The loader
verifies the summary, resolved tests, JSONL records, checksums, run state, and
recomputable aggregate counts before using any metric.

```bash
agentqa compare artifacts/baseline/runs/<id> artifacts/candidate/runs/<id> \
  --output artifacts/comparison.json
```

Cases are aligned by `test_case_id` and classified as `NEW_REGRESSION`, `FIXED`,
`PERSISTENT_FAILURE`, `STABLE`, `NEW`, or `REMOVED`. Removed cases remain in the
candidate effective pass-rate denominator. Deleting a failing or expensive test
therefore cannot silently improve the gate result; removals are also listed in
the comparison report.

The report includes pass rate, Mutation Score, high-risk failures, latency P50
and P95, tokens, cost, LLM/tool calls, retries, duplicate tool calls,
no-progress loops, replay hit rate, and verified failure-bundle coverage. Each
quantity retains its count, mean, P50, and P95 where applicable. A metric is
marked unobservable unless every trial provides the observation. Relative
growth from a zero baseline is marked unbounded instead of being coerced to
zero.

Duplicate tool calls are repeated calls with the same normalized tool name and
arguments inside one trace. A no-progress loop is a consecutive repeated LLM or
tool event with the same normalized input, output, and state diff. These are
diagnostic efficiency signals, not universal correctness claims.

`quality_cost` presents quality preservation and cost, token, and latency
improvement side by side. Cost is called improved only when average observed
cost decreases; the favorable
`QUALITY_PRESERVED_COST_IMPROVED` state additionally requires no effective
pass-rate regression, no increase in high-risk failures, and no new regression.
Tradeoffs and unobservable cost stay explicit. AgentQAKit deliberately does not
collapse quality and efficiency into an opaque combined score.

## QualityGate

```yaml
apiVersion: agentqa.dev/v1alpha1
kind: QualityGate
metadata:
  name: ci-regression
rules:
  minPassRate: 0.90
  minMutationScore: 0.80
  maxNewRegressions: 0
  maxHighRiskFailures: 0
  maxP95LatencyIncrease: 0.20
  maxAverageCostIncrease: 0.15
  requireReplayBundleForFailures: true
```

Evaluate it against a candidate and, for relative rules, a baseline:

```bash
agentqa gate evaluate quality-gate.yaml \
  --baseline artifacts/baseline/runs/<id> \
  --result artifacts/candidate/runs/<id> \
  --output artifacts/gate-result.json
```

Each configured rule reports its actual value, threshold, `PASS` or `FAIL`, and
reason. A configured but unobservable metric fails visibly. Exit code `0` means
the gate passed, `1` means one or more rules failed, and `2` means the input,
configuration, or system was invalid.

The repository workflow runs a deterministic local baseline and candidate,
writes the result to the GitHub Actions Job Summary, and always uploads the run
directories and checksummed reports. It deliberately does not post PR comments,
so it requires no write token and remains safe for forked pull requests.
