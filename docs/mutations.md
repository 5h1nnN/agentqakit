# Mutation testing

T5 injects a named, controlled defect at a real tool, retriever, or HTTP
boundary. An `AgentTest` opts into exactly one mutation with its `mutation`
field; a normal test without that field remains a pass-through run.

```yaml
apiVersion: agentqa.dev/v1alpha1
kind: Mutation
metadata:
  name: course-api-timeout
target:
  kind: tool
  name: query_course
when:
  callIndex: 1
  arguments:
    request.campus: west
  argumentExists: [request.courseId]
  testLabels:
    suite: resilience
  probability: 1.0
  mode: first
operation:
  type: tool_timeout
  delayMs: 100
```

Run a suite and load mutations from a separate file or directory:

```bash
agentqa test run tests.yaml --target target.yaml --mutations mutations.yaml
```

Tests, the target, and mutations may instead share one multi-document YAML
file. Mutation names are unique within a run and every `AgentTest.mutation`
reference must resolve.

## Matching and reproducibility

The matcher checks target kind and name, one-based `callIndex`, exact values at
restricted dotted/indexed argument paths, required argument paths, test labels,
probability, and `first`/`every` mode. All configured conditions must match.

Probability does not consume global random state. It is derived with SHA-256
from the trial seed, mutation ID, interception kind/name, and call index. The
same run seed and test layout therefore make the same decision. Use
`probability: 1.0` for normal regression and benchmark runs.

Every attempted interception emits a `MutationApplication` and a linked trace
event. Evidence contains the condition decisions, call index, whether behavior
actually changed, a non-reversible invocation/result summary, behavior metadata,
and trace/span IDs. A matched field deletion that cannot find its field records
`triggered: false`; it is not credited as a detected mutation.

## In-process operators

Tool operators:

- `tool_timeout` (and compatibility alias `timeout`) waits `delayMs`, then raises
  a dedicated injected timeout.
- `tool_exception` raises a dedicated injected runtime error.
- `invalid_json` executes the real call, then returns malformed JSON text.
- `missing_field` executes the real call, then deletes an existing field.
- `wrong_value` executes the real call, then replaces an existing field.
- `duplicate_call_result` reuses the first result on a later call.
- `stale_result` executes the current call but returns the prior result.

Retriever operators:

- `empty_retrieval` returns no documents.
- `irrelevant_document` returns one synthetic unrelated document.
- `indirect_prompt_injection` returns one harmless synthetic instruction that
  references only the fictional `CAMPUS_TEST_TOKEN`.

For LangGraph, expose a factory that accepts `agentqa_runtime` and wrap the
actual boundary once:

```python
def build_graph(agentqa_runtime=None):
    wrapped_tool = agentqa_runtime.wrap_tool(query_course, name="query_course")
    wrapped_retriever = agentqa_runtime.wrap_retriever(
        campus_retriever,
        name="campus_docs",
    )
    return build_with(wrapped_tool, wrapped_retriever)
```

Sync and async callables, common runnable methods, and LangChain
`StructuredTool` functions are supported. Without an active trial mutation the
wrappers call the original object unchanged.

## Fixed-upstream HTTP fault proxy

External HTTP tools can be placed behind the local proxy:

```text
Agent -> AgentQAKit fault proxy -> one fixed Tool API
```

```bash
agentqa mutation proxy \
  --upstream https://tool.example/api \
  --allow-path /course \
  --route-name campus.api \
  --mutations examples/specs/mutations.yaml \
  --cassettes artifacts/fault-proxy-cassettes \
  --mode record
```

The proxy binds only to loopback, uses the configured upstream host regardless
of the request, accepts only exact path allowlist entries, rejects traversal and
absolute-form targets, does not follow redirects, ignores environment proxies,
and enforces request/response size limits. It is deliberately not a forward or
open proxy.

The HTTP operators are delay (`timeout`), synthetic 5xx (`http_status`),
disconnect (`http_disconnect`), `invalid_json`, `missing_field`, and
`wrong_value`. Requests select a configured mutation with
`x-agentqa-mutation-id`. The HTTP target adapter also propagates test, trial,
trace, seed, label, and mutation headers so deterministic matching works across
the process boundary.

In `record` mode, pass-through responses are saved under a request signature.
Credential headers are omitted and JSON/text bodies are redacted before disk
write. Replay therefore returns the sanitized representation. In `replay` mode,
a cassette miss returns 502 and no network request is made. Proxy mutation
attempts are appended to `applications.jsonl` beside the cassettes.

## Mutation Score

Each mutation has one of four honest outcomes:

- `KILLED`: it triggered and at least one associated test detected a failure.
- `SURVIVED`: it triggered and all conclusive associated tests passed.
- `NOT_REACHED`: no application reported `triggered: true`.
- `INVALID`: trials were missing, no test was associated, or evaluation evidence
  was missing/inconclusive for an otherwise successful target.

Only killed and survived mutations enter the denominator:

```text
Mutation Score = killed / (killed + survived)
```

The report always displays `NOT_REACHED` and `INVALID` counts. It is written to
`mutation-score.json`, embedded in `summary.json` and `RunResult`, and grouped by
operation. Recompute it from checksummed resolved specs and JSONL results:

```bash
agentqa mutation score artifacts/runs/<run-id>
agentqa mutation score artifacts/runs/<run-id> --json
```

Evaluator `ERROR`, `SKIPPED`, `NOT_OBSERVABLE`, and `NEEDS_REVIEW` do not become
false kills. An injected exception that actually makes the target return an
error is a kill even though normal evaluators are then skipped.
