# Synthetic example agents

These agents are deliberately deterministic and use only synthetic data. They are test
subjects for AgentQAKit, not production assistants.

## Campus services agent

`campus_langgraph.py` exposes five graph factories:

| Version | Intentional behavior |
|---|---|
| `v0-bug1` | retries a timed-out course tool until the target timeout |
| `v0-bug2` | claims event registration success with missing required input |
| `v0-bug3` | calls the score tool before enforcing ownership |
| `v0-bug4` | follows an instruction embedded in retrieved content |
| `v1-fixed` | bounds retries, validates tool responses, authorizes before tools, and treats retrieval as untrusted data |

The graph provides `query_course`, `query_score`, `register_event`, `get_profile`, and a
small `campus_docs` retriever. Registration side effects live only in an in-memory set
owned by the graph instance.

## Open-source project assistant

`opensource_langgraph.py` is read-only. It searches a synthetic repository and local
documentation, queries synthetic issues, and mocks one allowlisted test command. No shell
command is executed.

Run a target through the benchmark script described in `benchmark/README.md`, or use its
target YAML directly with `agentqa test run`.

