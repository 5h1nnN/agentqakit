# CampusOps Agent

CampusOps Agent is AgentQAKit's complete, deliberately imperfect showcase target. It is a
small campus operations assistant with deterministic behavior, real tool/retriever boundaries,
authorization and privacy rules, a confirmed external write, idempotency, and one two-step task.
It runs without AgentQAKit and without an API key; AgentQAKit only wraps its boundaries when a
LangGraph target factory receives `agentqa_runtime`.

The project is synthetic. Every person, score, event, credential-like value, and endpoint is
fictional and safe to publish.

## Architecture

```text
request
  -> planner.py       deterministic rules or optional DeepSeek structured Action
  -> policies.py      validation, authorization, confirmation, privacy
  -> service.py       orchestration, retry, response validation, audit state
  -> tools.py         course/score/profile reads and idempotent registration write
  -> retriever.py     local policy retrieval with an untrusted-content boundary
  -> graph.py         thin LangGraph + AgentQAKit runtime adapter
```

`service.py` owns the business flow and has no AgentQAKit dependency. `graph.py` is the only
framework adapter. Data lives in `data/*.json`, and a `CampusOpsService` instance owns its
in-memory registration state.

## Capabilities

- Query courses, scores, and field-minimized profiles.
- Retrieve policy documents without treating document text as instructions.
- Register for an event only after explicit confirmation, with a stable idempotency key.
- Check a course time and register for a non-conflicting event as one composed task.
- Emit an observable plan, result, and ordered audit trail.

The default planner is deterministic. `--planner deepseek` uses the separate
`CAMPUS_AGENT_LLM_API_KEY`, `CAMPUS_AGENT_LLM_BASE_URL`, and `CAMPUS_AGENT_LLM_MODEL`
configuration. The model only produces an `Action`; all policy and tool-result decisions remain
deterministic.

## Run independently

From this directory:

```bash
python -m pip install -e '.[dev]'
campus-ops --input-json '{"action":"query_course","course_id":"AI101"}'
campus-ops --variant fixed '查询 AI101 课程时间'
campus-ops --variant fixed --input-json \
  '{"action":"course_then_register","course_id":"AI101","event_id":"event-robotics","confirmation":true}'
pytest
```

The deterministic service itself uses only the standard library. LangGraph is needed only for
`campus_ops_agent.graph` and is installed by the project package.

## Controlled variants

| Variant | Reproducible defect | Fixed invariant |
|---|---|---|
| `bug-retry` | retries a timed-out course tool without a bound | at most two attempts |
| `bug-confirmation` | calls `register_event` without confirmation | confirm before write |
| `bug-authorization` | calls `query_score` before denying access | authorize before tool |
| `bug-injection` | follows an injected retrieval instruction | retrieved text is data |
| `bug-idempotency` | creates another registration for a repeated key | stable deduplication |
| `fixed` | all known defects repaired | regression baseline |

Each switch is independent in `variants.py`. `agentqa/defect-registry.yaml` records its trigger,
root-cause symbol, violated policy, expected mutation, and Top-1/Top-3 failure-analysis truth.

## AgentQAKit assets

Install AgentQAKit from the repository root and run the fixed target:

```bash
python -m pip install -e '../../..[langgraph,mcp]'
agentqa test validate agentqa/golden-tests
agentqa test run agentqa/golden-tests \
  --target agentqa/targets/fixed.yaml \
  --manifest agentqa/manifest-v2.yaml
```

The `agentqa/` directory contains:

- `manifest-minimal.yaml`: T16 black-box condition A.
- `manifest-v2.yaml`: full semantic contract for condition B.
- `generation-goal-code.yaml`: the same goal plus selected CodeEvidence for condition C.
- six targets, eleven mutations, a quality gate, 19 human golden tests, and defect truth.

Run the separate mutation suite (its test failures are expected kills):

```bash
agentqa test run agentqa/mutation-tests \
  --target agentqa/targets/fixed.yaml \
  --mutations agentqa/mutations.yaml \
  --manifest agentqa/manifest-v2.yaml
```

`duplicate-registration-result` is intentionally retained as a `NOT_REACHED` control: the public
registration flow makes one write per turn, while cross-turn idempotency is asserted in
`tests/test_variants.py`. `NOT_REACHED` is reported honestly and excluded from Mutation Score.

Regenerate line ranges and hashes after changing relevant source:

```bash
python scripts/build_code_evidence.py
```

For the T16 comparison, keep model, temperature, token limit, goal, case limit, fixed target,
and random seed constant. Change only the manifest/evidence condition. Persist valid-case rate,
hallucinated tools, semantic coverage, defect detection, Mutation Score, token/cost data, review
actions, and analysis Top-1/Top-3 accuracy.

## Boundary and limitations

This target is intentionally a medium-short tool agent. It does not model long-horizon autonomy,
self-modification, or multi-agent collaboration. Registration is an in-memory synthetic write, so
tests are safe and deterministic. A real deployment would replace the data adapters and persist
idempotency records transactionally without changing the policy/orchestration contract.

