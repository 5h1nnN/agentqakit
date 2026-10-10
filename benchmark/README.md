# AgentQA-Bench

Regenerate and validate the versioned corpus:

```bash
python scripts/generate_benchmark.py
agentqa test validate benchmark/cases
agentqa test validate benchmark/mutations.yaml
```

Run every baseline and ablation, producing only script-derived evidence:

```bash
python scripts/run_benchmark.py \
  --config benchmark/config.yaml \
  --output benchmark/results/run-001
```

Add `--judge-provider deepseek` to enable the LLM Judge on the 20 Gold Cases. Without this
flag, judge evaluations are explicitly `SKIPPED`; they are never replaced by fabricated
scores. Use `--configurations output-only,output-trace` or `--limit 2` for a local smoke
run.

The output directory contains:

```text
raw_results.jsonl
summary.json
mutation_matrix.csv
metrics.csv
failures/
charts/
environment.json
```

It also keeps an `artifacts/` directory with the native AgentQAKit evidence used to derive
those files. Treat `raw_results.jsonl` and native artifacts as the source of truth; do not
edit result numbers or charts by hand.

