# Evaluation engine

AgentQAKit registers eight built-in evaluator types: `schema`, `text`,
`tool_call`, `trajectory`, `state`, `budget`, `privacy`, and `llm_judge`. The
first seven are deterministic. `llm_judge` is optional and includes a
fail-closed DeepSeek provider for live semantic evaluation.

## Result states

- `PASS`: all observable assertions passed.
- `FAIL`: target behavior contradicted at least one assertion.
- `ERROR`: the evaluator itself crashed, timed out, returned invalid data, or
  was not registered.
- `SKIPPED`: evaluation was intentionally not run, including a disabled Judge.
- `NOT_OBSERVABLE`: required evidence was absent or incomplete.
- `NEEDS_REVIEW`: the Judge explicitly requested human review.

A target `ERROR` or `TIMEOUT` still makes the trial fail. Built-in evaluators
may additionally run on that result to preserve useful assertion evidence.
`FAIL` takes precedence when both a violation and an unobservable metric exist;
the unknown metric remains recorded in metadata.

## Deterministic evaluators

`schema` checks `expect.output.jsonSchema` using JSON Schema Draft 2020-12,
`exactFields`, `requireNonEmpty`, and `allowedHttpStatuses`. `text` checks
required/forbidden patterns and deterministic refusal intent. Regex input is
bounded by `config.maxTextChars` (default 32 KiB), patterns are limited to 1024
characters, and every match has a timeout configured by
`config.patternTimeoutMs` (default 50 ms).

`privacy` applies only explicitly declared forbidden patterns and refusal
requirements. It does not claim heuristic PII detection. Extra patterns can be
placed in `evaluator.config.forbiddenPatterns`.

`tool_call` and `trajectory` require a `complete` trace. They operate on the
normalized chronological event sequence, report offending span IDs, and never
treat missing trace content as an empty argument object. If tool arguments were
retained only as a non-reversible summary, argument-schema evaluation returns
`NOT_OBSERVABLE`.

`state` also requires a complete trace. It reconstructs state by deep-merging
allowlisted `state_diff` values over `scenario.initialState`, then applies the
restricted non-executable DSL. `before` compares the first observed update of
two fields.

`budget` uses measured target latency and observed metadata or trace metrics.
Missing token, cost, retry, or trace-derived metrics remain unknown; they are
never replaced with zero.

## DeepSeek Judge

Live model calls are explicit. Copy `.env.example` to `.env.local`, fill the
three required values, and restrict the file permissions:

```bash
chmod 600 .env.local
agentqa test run examples/specs/ci-test.yaml \
  --target examples/specs/ci-candidate-target.yaml \
  --judge-provider deepseek \
  --no-trace-receiver
```

The DeepSeek provider uses Chat Completions JSON Output with a bounded token
limit, no redirects, one retry for transient transport/429/5xx failures, and
strict `JudgeDecision` validation. For credential safety, its Base URL must be
the official `https://api.deepseek.com` endpoint, optionally ending in `/v1`.
The API key is never placed in provider metadata, prompts, artifacts, database
records, error messages, or the reconstructed CLI command.

Before calling the provider, Judge selects a bounded, redacted evidence bundle.
It includes relevant Manifest purpose, capabilities, policies, and tool risk or
side-effect declarations; prior deterministic evaluator results; state diffs;
selected tool-call summaries and error spans; and explicit completeness gaps.
`manifestCapabilities` and `manifestPolicies` in evaluator `config` can select
named declarations. Unknown selections remain visible as missing evidence.
Judge is always scheduled after non-Judge evaluators, even if it appears first
in the YAML, while persisted evaluation order remains unchanged.

If a required deterministic result is missing, a required state transition is
not observable, or the trace is incomplete, Judge cannot turn a model `PASS`
into an unconditional pass. AgentQAKit changes it to `NEEDS_REVIEW` and records
the blocking gaps. Deterministic failures and high-risk deterministic
requirements remain authoritative.

The provider stores only safe accounting evidence: resolved model, request ID,
finish reason, system fingerprint, latency, use case, and token usage. Each
result also records provider/model/configuration, Rubric Hash, System Prompt
Hash, Prompt Hash, Evidence Hash, Response Schema Hash, status, and the number
of parse and transport attempts. A local request ID follows retries without
exposing credentials. Per-request output-token, timeout, and retry limits can
only tighten the configured provider limits. DeepSeek JSON mode can occasionally return empty content;
that is handled as invalid structured output and retried once by the evaluator.

Without `--judge-provider deepseek`, `llm_judge` returns `SKIPPED`, preserving
offline and deterministic CI. The local API exposes whether a Judge is
configured through `/api/health`; a run opts in with `judgeEnabled: true`.

## Custom providers

DeepSeek implements the provider-neutral `StructuredLLMProvider` used by
semantic judging, test generation, and failure analysis:

```python
from collections.abc import Mapping

from pydantic import JsonValue

from agentqa.evaluators import StructuredRequest, StructuredResponse, built_in_evaluators


class MyStructuredProvider:
    name = "my-provider"
    model = "my-model"
    parameters: Mapping[str, JsonValue] = {"temperature": 0.0}

    async def complete(self, request: StructuredRequest) -> StructuredResponse:
        # Send request.system_prompt, request.prompt, and request.response_schema.
        return StructuredResponse(
            content={"verdict": "PASS", "score": 1.0, "rationale": "Correct.", "violations": []}
        )


evaluators = built_in_evaluators(MyStructuredProvider())
```

The previous `JudgeProvider.judge()` contract remains accepted for backward
compatibility, but new integrations should implement `complete()`.

Every provider receives only the bounded, redacted evidence package, never an
implicit source tree or full raw trace. It must return JSON matching the
strict `JudgeDecision` schema with `PASS`, `FAIL`, or `NEEDS_REVIEW`. Invalid
JSON/schema output is retried once. Artifacts retain provider/model/parameters
plus prompt and rubric hashes, but not the raw prompt. Credential-like provider
parameters are redacted.

Tests labeled `risk: high` or `risk: critical` cannot configure only
`llm_judge`; at least one deterministic evaluator is required. A human-labeled
calibration dataset remains required before treating Judge scores as formal
effect metrics.

The separate, immutable `AnalysisReport` workflow for failed trials is
documented in [`failure-analysis.md`](failure-analysis.md).
