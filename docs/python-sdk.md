# Python SDK

The AgentQAKit Python SDK adds observable Agent, Tool, and Retriever boundaries
to an existing Python application. It is included in the `agentqakit`
distribution and imported as `agentqa`; there is no separate SDK package.

## Installation

Before the first PyPI release, install from a source checkout:

```bash
python -m pip install -e .
```

After a published release, install the pinned alpha explicitly:

```bash
python -m pip install "agentqakit==0.1.0a0"
```

Alpha APIs may change. Pin the exact version in reproducible environments.

## Decorate an Agent and its dependencies

```python
from typing import Any

import agentqa


@agentqa.tool(name="course.lookup")
def lookup_course(course_id: str) -> dict[str, str]:
    return {"course_id": course_id, "status": "open"}


@agentqa.retriever(name="policy.search")
async def search_policy(query: str) -> list[dict[str, str]]:
    return [{"title": "Registration policy", "text": query}]


@agentqa.agent(name="campus-agent")
async def run_agent(payload: dict[str, Any]) -> dict[str, Any]:
    course = lookup_course(str(payload["course_id"]))
    documents = await search_policy("registration")
    return {"status": "ok", "course": course, "documents": documents}
```

The decorators support synchronous and asynchronous callables and preserve the
wrapped callable's name, documentation, and signature metadata. Outside an
AgentQAKit trial they are transparent. During a trial they allow the runner to:

- record normalized Agent, Tool, and Retriever trace events;
- apply a declared mutation at the real dependency boundary;
- record or strictly replay dependency interactions; and
- evaluate tool order, call limits, trajectory, privacy, and budget rules.

Arguments and results are summarized through AgentQAKit's bounded evidence
layer. Applications must still avoid placing secrets in ordinary business
outputs and must follow their own data-retention requirements.

## Configure an isolated Python target

Point a Target specification at the decorated entrypoint:

```yaml
apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: campus-agent
type: python
entrypoint: my_agent.app:run_agent
isolation: subprocess
workerMode: perInvocation
timeoutMs: 5000
```

Validate and execute it with the ordinary CLI:

```bash
agentqa test validate tests/agentqa/
agentqa test run tests/agentqa/ \
  --target tests/agentqa/target.yaml \
  --manifest agent-manifest.yaml
```

The runnable reference is
[`examples/sdk_decorator_agent.py`](../examples/sdk_decorator_agent.py), with
matching Target, Test, and Mutation specifications under `examples/specs/`.

## Compatibility boundary

The stable import surface for the first alpha is:

```python
from agentqa import agent, retriever, tool
```

Internal runner, tracing, mutation, replay, workbench, and storage modules are
available to the CLI but are not yet declared stable SDK APIs. Review the
[changelog](../CHANGELOG.md) before upgrading any pre-1.0 release.
