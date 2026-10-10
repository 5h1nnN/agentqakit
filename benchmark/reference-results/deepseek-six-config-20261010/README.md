# DeepSeek six-configuration benchmark — 2026-10-10

This directory contains the reviewable, script-derived evidence from a complete
AgentQA-Bench baseline and ablation run. The full native run artifacts, traces,
cassettes, inputs, and reproduction ZIPs remain outside Git under
`benchmark/results/deepseek-six-config-20261010/`.

## Reproduction

```bash
python scripts/run_benchmark.py \
  --config benchmark/config.yaml \
  --output benchmark/results/deepseek-six-config-20261010 \
  --judge-provider deepseek
```

The run used the official DeepSeek endpoint configured through `.env.local`.
Secrets are not stored in these files. A post-run byte scan confirmed that the
configured API key is absent from both the native and reviewable result trees.

## Outcome

| Configuration | Passed | Failed | Pass rate | Judge PASS | Judge review | Judge error | Replay hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `output-only` | 110 | 0 | 100.0% | 0 | 0 | 0 | 0 |
| `output-judge` | 104 | 6 | 94.5% | 14 | 2 | 4 | 0 |
| `output-trace` | 64 | 46 | 58.2% | 0 | 0 | 0 | 0 |
| `full-agentqakit` | 90 | 20 | 81.8% | 15 | 0 | 5 | 0 |
| `no-replay-retry` | 61 | 49 | 55.5% | 15 | 0 | 5 | 0 |
| `with-replay` | 92 | 18 | 83.6% | 17 | 0 | 3 | 58 |

- Aggregate results: 660 (110 cases in each of six configurations)
- Real LLM Judge evaluations: 80
- Judge outcomes: 61 `PASS`, 2 `NEEDS_REVIEW`, 17 `ERROR`
- HTTP attempts: 105, all returned HTTP 200
- Resolved model: `deepseek-flash`
- Prompt tokens: 55,025
- Completion tokens: 72,548
- Total tokens: 127,573
- Native run artifacts: 122

The 17 `ERROR` outcomes were caused by invalid structured model output after one
retry, not by authentication or transport failures. The authoritative details
remain in `raw_results.jsonl`.

## Integrity

The authoritative raw-result hash recorded in `summary.json` is:

```text
sha256:e5541de83230d26d2b1aa8bcf2a4e9d53b9daeabf6fc0cbdf6c9f3cc2f17778d
```

Do not edit result numbers or the chart by hand. Regenerate them with the command
above.
