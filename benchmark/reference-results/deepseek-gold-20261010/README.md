# DeepSeek Gold benchmark — 2026-10-10

This directory contains the reviewable, script-derived evidence from the real-LLM
AgentQA-Bench run. The full native run artifacts, traces, cassettes, and reproduction ZIPs
remain outside Git under `benchmark/results/deepseek-gold-full-20261010/` and should be
published as a CI artifact or GitHub Release asset when needed.

## Reproduction

```bash
python scripts/run_benchmark.py \
  --config benchmark/config.yaml \
  --output benchmark/results/deepseek-gold-full-20261010 \
  --judge-provider deepseek \
  --configurations output-judge
```

The run used the official DeepSeek endpoint configured through `.env.local`. Secrets are
not stored in these files. A post-run scan confirmed that the API key was absent from the
complete result directory.

## Outcome

- Benchmark cases: 110
- Passed: 106
- Failed because of Judge status: 4
- Real LLM Judge evaluations: 20
- Judge outcomes: 16 `PASS`, 1 `NEEDS_REVIEW`, 3 `ERROR`
- HTTP attempts: 27, all returned HTTP 200
- Resolved model: `deepseek-flash`
- Prompt tokens: 14,145
- Completion tokens: 19,194
- Total tokens: 33,339

The three `ERROR` outcomes were caused by invalid structured model output after one retry,
not by authentication or transport failures. The `NEEDS_REVIEW` outcome reported
insufficient trace evidence for a prompt-injection case. See `raw_results.jsonl` for the
unaltered evaluator evidence.

## Integrity

The authoritative raw-result hash recorded in `summary.json` is:

```text
sha256:33534799701ab28d2c3a5d22592d25ef414ba93039e9dc234f9d91a661a7bedc
```

Do not edit result numbers or the chart by hand. Regenerate them with the command above.
