"""Target adapter contract consumed by the test runner."""

from typing import Protocol

from agentqa.models import RunContext, TargetResult, TestInput


class TargetAdapter(Protocol):
    """Prepare, invoke, and clean up one configured agent target."""

    async def prepare(self, context: RunContext) -> None: ...

    async def invoke(self, test_input: TestInput, context: RunContext) -> TargetResult: ...

    async def cleanup(self, context: RunContext) -> None: ...
