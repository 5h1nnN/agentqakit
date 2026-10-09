# Evaluation engine

AgentQAKit registers eight built-in evaluator types: `schema`, `text`,
`tool_call`, `trajectory`, `state`, `budget`, `privacy`, and `llm_judge`. The
first seven are deterministic. `llm_judge` is optional and has no built-in
network provider.

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

## Provider-neutral Judge

The CLI intentionally constructs `llm_judge` without a provider, so it returns
`SKIPPED`. Applications can inject a provider through the public protocol:

```python
from collections.abc import Mapping

from pydantic import JsonValue

from agentqa.evaluators import JudgeRequest, built_in_evaluators


class MyJudgeProvider:
    name = "my-provider"
    model = "my-model"
    parameters: Mapping[str, JsonValue] = {"temperature": 0.0}

    async def judge(self, request: JudgeRequest) -> str:
        # Send request.prompt and request.response_schema using your own client.
        ...


evaluators = built_in_evaluators(MyJudgeProvider())
```

The provider receives only bounded, redacted scenario/output evidence and a
summary of trace tool names and error count. It must return JSON matching the
strict `JudgeDecision` schema with `PASS`, `FAIL`, or `NEEDS_REVIEW`. Invalid
JSON/schema output is retried once. Artifacts retain provider/model/parameters
plus prompt and rubric hashes, but not the raw prompt. Credential-like provider
parameters are redacted.

Tests labeled `risk: high` or `risk: critical` cannot configure only
`llm_judge`; at least one deterministic evaluator is required. A human-labeled
calibration dataset belongs to the provider/application layer because this
repository does not ship a model or make live model calls.
