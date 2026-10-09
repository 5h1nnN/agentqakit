# Tracing and instrumentation

T4 gives every trial one OpenTelemetry trace rooted at `agentqa.test_run`.
The root carries the run, test, trial, and optional mutation identifiers. SDK
spans, LangGraph v2 events, and remote OTLP spans are normalized to the same
`TraceEvent` model and written to `traces/<trace-id>.json` in the run artifact.

## Trace completeness

After a target response the runner waits briefly for late spans, then assigns
one explicit state:

- `complete`: one runner root exists and every observed target span has a known
  parent;
- `partial`: spans exist, but the root is ambiguous, parents are missing, a
  span did not end, or a collection limit was reached;
- `missing`: only the runner root was observed, so target internals are not
  observable.

Output evaluators can still inspect a successful target response when tracing
is missing. Evaluators declaring `requires_trace = True` receive
`NOT_OBSERVABLE` unless the trace is complete.

## Local OTLP receiver

`agentqa test run` starts a loopback-only OTLP/HTTP protobuf receiver at
`http://127.0.0.1:4318/v1/traces` by default. Change the port with
`--otlp-port`, or use `--no-trace-receiver` when another collector owns that
port. The receiver enforces body and per-trace span limits and returns a safe
client error for malformed protobuf.

An instrumented service can export to the receiver with the standard settings:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_TRACES_EXPORTER=otlp
```

## Zero-source-change instrumentation

Install optional OpenAI and framework instrumentors:

```bash
python -m pip install -e ".[instrumentation]"
```

Then launch an agent through the wrapper:

```bash
agentqa instrument \
  --framework langgraph \
  --exporter http://127.0.0.1:4318 \
  -- uvicorn my_agent:app
```

The command selects installed OTel instrumentors, configures OTLP/HTTP, forwards
termination signals, and returns the child process exit code. It does not edit
application source. OpenInference is preferred for OpenAI calls; the in-process
integration has a bounded SDK wrapper fallback when OpenInference is absent.

## Privacy and limits

By default, prompt and response values become non-reversible summaries with a
type, byte length, and SHA-256 digest. Credential-like attributes, including
API keys and authorization headers, are always redacted. Enable
`--capture-trace-content` for local runs or `--capture-content` for the launcher
only when content retention is intentional; captured values remain redacted and
size-limited. Span count, attribute count, attribute length, content size, and
OTLP request size all have explicit bounds.
