# Evidence-grounded failure analysis

AgentQAKit can ask a structured LLM to explain one failed trial while keeping
the recorded test result and trace immutable. The output is an
`AnalysisReport`, not a replacement verdict: observed facts, the earliest
trajectory anomaly, ranked root-cause candidates, evidence still needed,
suggested regression tests, and optional repair directions remain explicitly
separate from the original `EvaluationResult`.

## Run an analysis

The source run must be complete and checksummed, the selected `trial_id` must
identify exactly one failed trial, and its trace and optional resolved Manifest
must pass checksum and schema validation.
Failure analysis requests up to 8192 output tokens; keep
`AGENTQA_LLM_MAX_TOKENS=8192` (as shown in `.env.example`) so the provider-level
cap does not truncate the structured report.

```bash
agentqa analyze failure \
  --run artifacts/runs/<run-id> \
  --trial-id <failed-trial-id> \
  --provider deepseek \
  --llm-env-file .env.local \
  --output artifacts/analysis/<trial-id>.json
```

The output path must remain inside `--workspace` (the current directory by
default) and must not already exist. AgentQAKit writes an adjacent `.sha256`
file. It does not modify the source run, trace, failure bundle, or test result.

## Evidence boundary

Judge and failure analysis share the same deterministic evidence selector. It
builds a redacted package capped at 256 KiB from:

- the bounded scenario and normalized target result;
- relevant Manifest purpose, capability, policy, and tool declarations;
- tool side-effect, risk, idempotence, and confirmation semantics;
- already-computed deterministic evaluator results;
- state snapshots and diffs, selected tool argument/result summaries, and
  error spans;
- trace completeness and an explicit list of missing or blocking evidence.

No source files or complete raw trace are sent implicitly. Each item has a
stable evidence ID and content hash. The bundle hash is deterministic for the
same selected evidence. Every fact, divergence, root-cause hypothesis, and
regression-test suggestion returned by the model must cite known evidence IDs;
unknown references fail schema validation. Invalid structured output receives
one repair attempt and then fails closed.

The report is non-authoritative. Confidence belongs to each root-cause
candidate, missing evidence remains visible, and repair directions are advice
only: the analysis service never edits the repository.

## Privacy and accounting

Credential-like keys and text are redacted before model input. Persisted LLM
evidence contains safe provider/model identity, configuration and prompt/schema
hashes, request identifiers, attempts, latency, and token usage. It never
contains the API key or raw prompt.
