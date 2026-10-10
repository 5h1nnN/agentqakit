# Test runner

The T2 runner executes validated `AgentTest` specifications against injected
`TargetAdapter` and `Evaluator` implementations. It deliberately does not know
whether an agent is reached through HTTP, Python, or a framework adapter.

## Lifecycle

The persisted lifecycle is:

```text
CREATED -> PREPARING -> RUNNING -> EVALUATING -> FINALIZING
                                                        |-> COMPLETED
                                                        |-> FAILED
                                                        `-> CANCELLED
```

Preparation failures fail the run. Invocation and evaluator failures become
structured trial results so later trials still execute. Cleanup is attempted
after every preparation attempt and has its own timeout. `summary.json` is
written last.

Each evaluator receives one immutable `EvaluationRequest` containing the
validated `AgentTestSpec`, its own `EvaluatorSpec`, the complete `TargetResult`,
the explicit `TraceRecord`, and the trial `RunContext`. An assertion mismatch is
`FAIL`; an evaluator crash, timeout, invalid result, or missing implementation is
`ERROR`. This distinction keeps product regressions separate from test
infrastructure defects.

## Trial policy

Multiple trials are reduced to one test-case decision with `trialPolicy`:

```yaml
trials: 3
trialPolicy:
  strategy: minPassRate
  minPassRate: 0.67
```

Supported strategies are:

- `all`: every trial must pass; this is the default.
- `majority`: strictly more than half of the trials must pass.
- `minPassRate`: the completed trial pass rate must meet `minPassRate`.

A case with both passing and failing trials is marked `inconsistent`. Run-level
`pass_rate` is the number of passing aggregated test cases divided by all
planned test cases.

The returned process-oriented exit code is `0` when every aggregated case
passes, `1` when execution completes with test failures, and `2` for an
infrastructure/specification failure, cancellation, or incomplete run.

## Reproducibility and metrics

Every trial gets a UUID trial ID, UUID trace ID, and deterministic seed derived
from the run seed, test name, and one-based trial index. The summary contains
mean, P50, and P95 latency, token count, and step count. Missing token or step
metadata remains unknown and is not converted to zero.

Wrapped dependency calls are recorded by default into one canonical cassette
per trial. `TestResult` links the cassette, carries replay hit/miss statistics,
and links a verified reproduction bundle when the trial fails. Bundle replay
restores the recorded trial seed instead of deriving a new one. See
[`replay.md`](replay.md).

## Cancellation and timeouts

`TestRunner.cancel()` cooperatively cancels an active invocation or evaluator,
runs cleanup, persists completed results, and returns a `CANCELLED` result.
Preparation, target invocation, each evaluator, the total test item, and cleanup
all have explicit timeouts.

## Concurrency and global budgets

`--concurrency N` bounds active trials with an `asyncio.Semaphore`; the default
of `1` preserves sequential execution. Trial, trace, replay, and mutation state
is isolated with context-local values, and returned results retain planned test
and trial order even when completion order differs.

Two optional run-wide limits stop new trials from being created:

```text
--max-total-tokens 100000
--max-total-cost-usd 5.00
```

The scheduler updates consumption after completed trials. Once either observed
total reaches its limit, already-running trials finish normally and unscheduled
trials remain incomplete, producing exit code `2`. Consequently, final use may
exceed a limit by the consumption of in-flight work. Token and cost values come
from target metadata or complete LLM trace evidence. Missing observations stay
unknown, do not consume budget, and are counted explicitly in the run's budget
summary rather than being invented as zero.

The built-in evaluators and their evidence semantics are described in
[`evaluators.md`](evaluators.md).
