"""Bounded in-process orchestration over persistent workflow checkpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from pydantic import JsonValue

from agentqa.errors import AgentQAError
from agentqa.models import ErrorInfo
from agentqa.workbench.models import ArtifactRef, WorkflowRun, WorkflowStep, WorkflowStepKind
from agentqa.workbench.store import WorkflowStore, WorkflowStoreError

WorkflowStepHandler: TypeAlias = Callable[[WorkflowStep], Awaitable[list[ArtifactRef]]]

_AI_STEPS: frozenset[WorkflowStepKind] = frozenset(
    {"GENERATE_PLAN", "GENERATE_TESTS", "ANALYZE_FAILURE"}
)
_CAMPAIGN_STEPS: frozenset[WorkflowStepKind] = frozenset(
    {"RUN_CAMPAIGN", "RUN_MUTATIONS", "RECORD_REPLAY"}
)


class WorkflowOrchestrator:
    """Schedule bounded work while persisting every externally visible transition."""

    def __init__(
        self,
        store: WorkflowStore,
        *,
        max_concurrent_ai_steps: int = 1,
        max_concurrent_campaigns: int = 1,
    ) -> None:
        if max_concurrent_ai_steps < 1 or max_concurrent_campaigns < 1:
            raise ValueError("workflow concurrency limits must be positive")
        self._store = store
        self._ai_semaphore = asyncio.Semaphore(max_concurrent_ai_steps)
        self._campaign_semaphore = asyncio.Semaphore(max_concurrent_campaigns)
        self._general_semaphore = asyncio.Semaphore(1)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def _semaphore(self, kind: WorkflowStepKind) -> asyncio.Semaphore:
        if kind in _AI_STEPS:
            return self._ai_semaphore
        if kind in _CAMPAIGN_STEPS:
            return self._campaign_semaphore
        return self._general_semaphore

    async def submit(
        self,
        *,
        workflow_id: str,
        kind: WorkflowStepKind,
        idempotency_key: str,
        input_data: dict[str, JsonValue],
        handler: WorkflowStepHandler,
        complete_workflow: bool = False,
    ) -> WorkflowStep:
        """Persist and schedule one idempotent workflow step."""
        step = await self._store.create_step(
            workflow_id=workflow_id,
            kind=kind,
            idempotency_key=idempotency_key,
            input_data=input_data,
        )
        async with self._lock:
            if step.step_id in self._tasks:
                return step
            if step.status != "QUEUED":
                return step
            self._tasks[step.step_id] = asyncio.create_task(
                self._execute(step, handler, complete_workflow=complete_workflow),
                name=f"agentqa-workflow-{step.step_id}",
            )
        return step

    async def _execute(
        self,
        step: WorkflowStep,
        handler: WorkflowStepHandler,
        *,
        complete_workflow: bool,
    ) -> None:
        try:
            async with self._semaphore(step.kind):
                running = await self._store.start_step(step.step_id)
                artifacts = await handler(running)
                await self._store.finish_step(
                    running.step_id,
                    artifacts=artifacts,
                    complete_workflow=complete_workflow,
                )
        except asyncio.CancelledError:
            workflow = await self._store.get_workflow(step.workflow_id)
            if workflow is not None and workflow.status != "CANCELLED":
                await self._store.interrupt_step(
                    step.step_id,
                    ErrorInfo(
                        code="WORKFLOW_STEP_INTERRUPTED",
                        message="workflow step was interrupted before completion",
                    ),
                )
            raise
        except AgentQAError as exc:
            await self._fail_if_running(step.step_id, ErrorInfo(code=exc.code, message=exc.message))
        except Exception as exc:
            await self._fail_if_running(
                step.step_id,
                ErrorInfo(
                    code="WORKFLOW_STEP_ERROR",
                    message=f"workflow step raised {type(exc).__name__}",
                ),
            )
        finally:
            async with self._lock:
                self._tasks.pop(step.step_id, None)

    async def _fail_if_running(self, step_id: str, error: ErrorInfo) -> None:
        try:
            await self._store.fail_step(step_id, error)
        except WorkflowStoreError as exc:
            if exc.code != "WORKFLOW_STEP_TRANSITION_ERROR":
                raise

    async def cancel(self, workflow_id: str) -> WorkflowRun:
        """Cancel queued/running work without deleting prior evidence."""
        workflow = await self._store.cancel_workflow(workflow_id)
        async with self._lock:
            tasks = [
                task
                for step_id, task in self._tasks.items()
                if any(step.step_id == step_id for step in workflow.steps)
            ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        refreshed = await self._store.get_workflow(workflow_id)
        return workflow if refreshed is None else refreshed

    async def resume(self, workflow_id: str) -> WorkflowRun:
        """Return an interrupted/draft workflow to READY for a new step attempt."""
        return await self._store.resume_workflow(workflow_id)

    async def wait(self, step_id: str) -> WorkflowStep | None:
        """Wait for a locally active step and return its persisted state."""
        async with self._lock:
            task = self._tasks.get(step_id)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        for workflow in await self._store.list_workflows():
            match = next((step for step in workflow.steps if step.step_id == step_id), None)
            if match is not None:
                return match
        return None

    async def shutdown(self) -> None:
        """Interrupt active local tasks so a later process can resume safely."""
        async with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["WorkflowOrchestrator", "WorkflowStepHandler"]
