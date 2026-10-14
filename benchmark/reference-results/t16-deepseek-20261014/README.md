# T16 DeepSeek end-to-end evidence — 2026-10-14

This directory is the reviewable subset of one real DeepSeek T16 run. Full traces, cassettes,
failure ZIPs, generated YAML, and intermediate checkpoints remain outside Git under
`benchmark/results/t16-deepseek-20261014/` and
`benchmark/results/t16-mcp-deepseek-20261014/`.

## Reproduction

```bash
python scripts/run_t16.py \
  --output benchmark/results/t16-deepseek-20261014 \
  --approve-high-risk \
  --max-generated-cases 6 \
  --llm-max-tokens 16384 \
  --llm-timeout-seconds 180

python scripts/run_t16_mcp.py \
  --output benchmark/results/t16-mcp-deepseek-20261014 \
  --approval-id t16-human-review-20261014 \
  --max-generated-cases 4 \
  --llm-max-tokens 16384 \
  --llm-timeout-seconds 180
```

The run used the official DeepSeek endpoint configured through the ignored `.env.local` file.
No prompt, raw response, repository dump, or API key is included here. A post-run byte scan found
none of the configured key, token, or secret values in either final raw result tree.

## Results

The deterministic human-authored baseline passed all 21 tests. Its regression gate passed with
no new regression, and the mutation campaign killed all 10 reached mutations; one mutation was
`NOT_REACHED`, so the denominator-based Mutation Score was 100% and the mutation gate passed.
The standalone Judge path made one real model call.

The controlled generation comparison used the same six-case goal and provider parameters in all
three arms:

| Arm | Strictly valid tests | Generation status | Estimated generation cost |
| --- | ---: | --- | ---: |
| Minimal Manifest | 0/6 | `FAILED_SAFE` after one repair | $0.01368561 |
| Manifest v2 | 0/6 | `FAILED_SAFE` after one repair | $0.01622165 |
| Manifest v2 + seven CodeEvidence excerpts | 6/6 | `COMPLETED` | $0.00763598 |

This single run supports the usefulness of selected code context, but it is not a statistical
claim. In the successful arm, five behavior tests and one mutation test were generated without
manual edits. Four of the five behavior tests passed the fixed Agent; the composition case made
an incorrect non-conflicting-event assumption. The suite detected 2/5 seeded defects
(authorization and missing confirmation), and its generated-only Mutation Score was 0%. These
limitations are retained rather than filtered from the result.

The AI failure-analysis benchmark completed all five held-out defect cases with no analysis
failure. Term matching against the held-out truth registry produced Top-1 5/5 and Top-3 5/5, at
an estimated cost of $0.02270069.

The independent MCP collaboration path generated and validated 4/4 tests, observed three failures
in the 21-case broken run, reran the fixed Agent at 21/21, executed the full mutation campaign,
reported three fixed cases and zero new regressions, and passed both quality gates. Four generated
cases are used here because that strict structured transaction was stable in the verified run;
this budget does not reduce the human golden or mutation suites.

## Files and integrity

- `summary.json` and `metrics.csv`: experiment-level results and compact arm comparison.
- `environment.json`: model parameters, seed, platform, source hash, and captured pricing snapshot.
- `plan.json` and `generation-report.json`: successful CodeEvidence planning/generation evidence.
- `scores.json`: held-out failure-analysis scores and token totals.
- `regression-gate.json` and `mutation-gate.json`: standalone quality-gate evidence.
- `mcp-transcript.json`: official MCP resource/tool golden-path transcript.

Every data file has an adjacent `.sha256` sidecar. Verify the directory with:

```bash
for sidecar in benchmark/reference-results/t16-deepseek-20261014/*.sha256; do
  target=${sidecar%.sha256}
  expected=$(tr -d '[:space:]' < "$sidecar")
  actual="sha256:$(shasum -a 256 "$target" | cut -d ' ' -f 1)"
  test "$expected" = "$actual" || exit 1
done
```

Do not edit measured values by hand. Regenerate the raw run, recheck its data boundary and hashes,
then replace this reviewable subset.
