"""Python callable fixtures exercised in worker subprocesses."""

import asyncio
from typing import Any, cast

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
