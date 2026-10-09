# Record, replay, and failure bundles

AgentQAKit records wrapped tool and retriever interactions by default. Each
trial owns one canonical JSONL cassette under `cassettes/`; failed trials also
produce a checksummed reproduction ZIP under `failures/`.

```bash
# Default policy: call live dependencies and record them.
agentqa test run tests.yaml --target target.yaml

# Strict replay: every wrapped dependency call must match the source cassette.
agentqa test run tests.yaml --target target.yaml \
  --replay-mode replay \
  --replay-cassette artifacts/runs/<run-id>/cassettes/<trial-id>.jsonl
```

Use `--replay-mode off` only when dependency evidence is intentionally not
needed. `--allow-live-fallback` is valid only in replay mode. It turns a miss
into a live call and writes the observed delta, so it should be used for
cassette maintenance rather than deterministic CI reproduction.

## Matching and call order

Each entry is keyed by:

```text
kind + logical name + canonical arguments + one-based call index
```

Object keys are sorted. Tool and retriever wrappers assign call indices before
invocation and persist entries at request completion, including safe exception
evidence. Repeated identical calls remain separate. Replay returns recorded
results, raises a safe `ReplayedDependencyError` for recorded exceptions, and
fails closed with `CASSETTE_MISS` when no entry matches.

For LangGraph targets, factories opt in to the boundary by accepting
`agentqa_runtime` and wrapping real tools or retrievers:

```python
def build_graph(agentqa_runtime=None):
    tool = agentqa_runtime.wrap_tool(query_course, name="query_course")
    retriever = agentqa_runtime.wrap_retriever(course_docs, name="course_docs")
    return build_with(tool, retriever)
```

Mutation interception is outside record/replay interception. A cassette thus
stores the baseline dependency response, while a mutation changes the value
seen by the agent. Mutations that short-circuit before a dependency call do not
create a cassette entry.

## Privacy and integrity

Credential-shaped keys and credential text are redacted before entry checksums
or artifact hashes are computed. Declare nondeterministic request fields with a
repeatable option; they are removed before matching:

```bash
agentqa test run tests.yaml --target target.yaml \
  --volatile-field request_id \
  --volatile-field metadata.timestamp
```

Large results are written to content-addressed `blobs/*.json` files with SHA-256
sidecars. JSONL entries reference those blobs. Cassette entries also contain
the trace and span IDs emitted by the wrapper, and `TestResult.replay` records
hits, misses, live fallbacks, and hit rate.

The fixed-upstream HTTP fault proxy uses the same cassette schema at
`<cassette-directory>/cassette.jsonl`. Replay misses return 502 without network
access unless its own `--allow-live-fallback` flag is supplied.

## Reproducing a failed trial

Every failed trial creates `failure-<trial-id>.zip` by default. Disable this
only when required with `--no-failure-bundles`. A bundle contains the resolved
test and target, optional mutation and agent manifest, result, trace, cassette
and blobs, environment metadata, instructions, and checksums.

```bash
agentqa replay artifacts/runs/<run-id>/failures/failure-<trial-id>.zip
```

The replay command validates every checksum and rejects duplicate, absolute,
traversal, symlink, oversized, or inconsistent members before extracting into
a new temporary directory. It then runs exactly one strict-replay trial with
the recorded seeds. Success means the original failure signature was
reproduced; exit `1` means execution completed but the signature changed, and
exit `2` means the bundle or replay infrastructure was invalid.

Bundles intentionally do not copy application source or install dependencies.
The target entrypoint and its Python environment must be available on the
machine performing the replay.
