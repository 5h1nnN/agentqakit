# Target adapters

T3 connects the sequential runner to HTTP services, isolated Python callables,
and LangGraph-compatible graphs. All adapters implement the same
`prepare -> invoke -> cleanup` protocol and return structured `TargetResult`
objects instead of leaking target exceptions into the runner.

## Running a target

Pass a test file or directory and a path containing exactly one target:

```bash
agentqa test run examples/specs/basic-test.yaml \
  --target examples/specs/http-target.yaml
```

The command writes the normal run artifacts and exits with the runner's exit
code. The CLI registers AgentQAKit's built-in T6 evaluators. A trial passes only
when the target succeeds and every configured evaluator returns `PASS`.

## HTTP

HTTP targets support GET and POST JSON requests, restricted `input` templates,
query parameters, fixed headers, response output paths, and an allowlist of
response headers to retain. Every request receives W3C `traceparent` plus run,
test, trial, and trace identity headers. Active mutation ID, deterministic trial
seed, and test labels are included when available for fault-proxy matching.

```yaml
apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: local-http
type: http
url: http://localhost:8000/chat
method: POST
request:
  headers:
    content-type: application/json
  query:
    suite: regression
  json:
    message: "{{ input.message }}"
response:
  outputPath: $.answer
  allowedHeaders: [x-request-id]
connectTimeoutMs: 5000
readTimeoutMs: 30000
timeoutMs: 30000
maxResponseBytes: 1048576
```

The default network policy allows loopback only. Public remote endpoints need
`allowRemote: true`. Private-network endpoints additionally need
`allowPrivateNetwork: true`. `allowedHosts` can narrow either policy to exact
hostnames. Link-local, multicast, reserved, unspecified, and known cloud
metadata endpoints are always rejected. Redirects are never followed.

The adapter streams the response under a decoded-byte limit. Empty or non-JSON
responses, missing output paths, HTTP 3xx/4xx/5xx statuses, disconnects, and
timeouts receive stable error codes. Only explicitly allowed response headers
are retained, and credential-like header names are redacted.

## Python callable

A Python target is a `module:function` callable accepting one normalized
dictionary. The dictionary contains `test_case_id`, `input`, `initial_state`,
`messages`, and `mutation`. It may return any finite JSON-serializable value and
may be synchronous or asynchronous.

```yaml
apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: local-python
type: python
entrypoint: my_agent:run
isolation: subprocess
workerMode: perInvocation
environmentAllowlist: [MY_AGENT_SETTING]
timeoutMs: 10000
```

`perInvocation` is the default and starts a fresh interpreter for every trial,
so module globals cannot leak between cases. `persistent` reuses one worker for
the run when startup cost is important; user code must then manage its own
global state.

The parent and worker exchange correlated JSONL messages. User `print()` output
is redirected away from protocol stdout, captured with a fixed size limit, and
credential patterns are redacted. Workers receive only allowlisted environment
variables plus AgentQAKit run, test, trace, mutation, and artifact identifiers.
Timeout or cancellation terminates only the worker's dedicated process group.

## LangGraph

LangGraph is an optional integration:

```bash
python -m pip install -e ".[langgraph]"
```

An entrypoint can expose an already compiled graph or a factory:

```python
def build_graph(agentqa_runtime=None):
    tool = agentqa_runtime.wrap_tool(tool) if agentqa_runtime else tool
    return builder.compile()
```

```yaml
apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: graph-agent
type: langgraph
entrypoint: my_graph:build_graph
inputKey: messages
stateDiffAllowlist: [phase, retry_count]
```

The adapter supports sync and async invocation and consumes the v2 event stream
when the graph exposes `astream_events`. Chain, chat-model/LLM, tool, and
retriever events become `AGENT`, `NODE`, `LLM`, `TOOL`, and `RETRIEVER` trace
events. State diffs retain only configured keys. A compiled graph can run and
be observed, but its capability metadata recommends a factory when complete
tool or retriever wrapping cannot be guaranteed.

The `AgentQARuntime` wrapper methods apply T5 tool and retriever mutations at
the real invocation boundary and emit linked application/trace evidence. T4
records framework events directly in the trial TraceStore; see
[`tracing.md`](tracing.md) for persistence, completeness, OTLP export, and
privacy behavior, and [`mutations.md`](mutations.md) for wrapper and mutation
semantics.
