# Sequential test runner

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

## Cancellation and timeouts

`TestRunner.cancel()` cooperatively cancels an active invocation or evaluator,
runs cleanup, persists completed results, and returns a `CANCELLED` result.
Preparation, target invocation, each evaluator, the total test item, and cleanup
all have explicit timeouts.

## Current scope

T2 is sequential by design. Concurrency, semaphores, and global Token/cost
budgets belong to the deferred T2-3 task. The runner's explicit per-trial
contexts and stable execution ordering provide the boundary for adding them
later.

The built-in evaluators and their evidence semantics are described in
[`evaluators.md`](evaluators.md).
