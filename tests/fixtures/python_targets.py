"""Python callable fixtures exercised in worker subprocesses."""

import asyncio
from typing import Any, cast

import agentqa

_calls = 0


def echo(value: dict[str, Any]) -> dict[str, Any]:
    return {"received": value["input"]}


async def async_echo(value: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(0)
    return {"received": value["input"]}


def prints(value: dict[str, Any]) -> dict[str, Any]:
    print("worker user log Authorization: secret-value")
    return cast(dict[str, Any], value["input"])


def nonserializable(value: dict[str, Any]) -> set[str]:
    del value
    return {"not-json"}


def raises(value: dict[str, Any]) -> None:
    del value
    raise RuntimeError("fixture failure")


async def waits_forever(value: dict[str, Any]) -> None:
    del value
    await asyncio.Event().wait()


def counter(value: dict[str, Any]) -> dict[str, Any]:
    del value
    global _calls
    _calls += 1
    return {"calls": _calls}


@agentqa.tool(name="sdk.lookup")
def sdk_lookup(message: str) -> dict[str, Any]:
    return {"message": message, "status": "available"}


@agentqa.retriever(name="sdk.documents")
async def sdk_documents(query: str) -> list[dict[str, str]]:
    await asyncio.sleep(0)
    return [{"page_content": f"document for {query}"}]


@agentqa.agent(name="decorated-agent")
async def decorated_agent(value: dict[str, Any]) -> dict[str, Any]:
    message = str(value["input"]["message"])
    result = sdk_lookup(message)
    documents = await sdk_documents(message)
    return {**result, "documents": documents}


@agentqa.tool(name="sdk.replay-dependency")
def record_dependency(query: str) -> dict[str, str]:
    return {"query": query, "source": "live"}


@agentqa.agent
def record_agent(value: dict[str, Any]) -> dict[str, str]:
    return record_dependency(str(value["input"]["message"]))


@agentqa.tool(name="sdk.replay-dependency")
def unavailable_dependency(query: str) -> dict[str, str]:
    del query
    raise RuntimeError("live dependency must not run during replay")


@agentqa.agent
def replay_agent(value: dict[str, Any]) -> dict[str, str]:
    return unavailable_dependency(str(value["input"]["message"]))
