import asyncio
from pathlib import Path

import pytest

from agentqa.conventions import utc_now
from agentqa.storage import SQLiteResultStore
from agentqa.workbench import (
    ApprovalDecision,
    ArtifactRef,
    SQLiteWorkflowStore,
    WorkflowOrchestrator,
    WorkflowStoreError,
)


def test_workflow_store_persists_idempotent_projects_workflows_and_events(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        store = SQLiteWorkflowStore(tmp_path / "workbench.sqlite3")
        await store.initialize()

        project = await store.create_project("campus", "Campus Ops", "Local showcase")
        assert await store.create_project("campus", "Campus Ops", "Local showcase") == project
        with pytest.raises(WorkflowStoreError, match="different configuration"):
            await store.create_project("campus", "Changed", "Local showcase")

        workflow = await store.create_workflow(
            project_id="campus",
            name="Regression review",
            description="Review the repaired candidate.",
            idempotency_key="workflow-request-1",
        )
        repeated = await store.create_workflow(
            project_id="campus",
            name="Regression review",
            description="Review the repaired candidate.",
            idempotency_key="workflow-request-1",
        )
        assert repeated.workflow_id == workflow.workflow_id
        assert workflow.status == "READY"
        assert workflow.revision == 1

        with pytest.raises(WorkflowStoreError) as conflict:
            await store.create_workflow(
                project_id="campus",
                name="Different workflow",
                description="Review the repaired candidate.",
                idempotency_key="workflow-request-1",
            )
        assert conflict.value.code == "WORKFLOW_IDEMPOTENCY_CONFLICT"

        events = await store.list_events(workflow.workflow_id)
        assert [event.event_type for event in events] == ["WORKFLOW_CREATED"]
        assert await store.list_events(workflow.workflow_id, after_revision=1) == []
        assert (await store.list_projects())[0].project_id == "campus"
        assert (await store.list_workflows("campus"))[0].workflow_id == workflow.workflow_id

    asyncio.run(exercise())


def test_workflow_and_result_stores_share_one_versioned_database(tmp_path: Path) -> None:
    async def exercise() -> None:
        database = tmp_path / "shared.sqlite3"
        result_store = SQLiteResultStore(database)
        workflow_store = SQLiteWorkflowStore(database)
        await result_store.initialize()
        await workflow_store.initialize()
        await workflow_store.create_project("shared", "Shared database")
        assert (await workflow_store.list_projects())[0].project_id == "shared"

    asyncio.run(exercise())


def test_workflow_restart_marks_running_state_interrupted_and_can_resume(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        database = tmp_path / "restart.sqlite3"
        first = SQLiteWorkflowStore(database)
        await first.initialize()
        await first.create_project("campus", "Campus Ops")
        workflow = await first.create_workflow(
            project_id="campus",
            name="Interrupted workflow",
            description="",
            idempotency_key="workflow-restart",
        )
        step = await first.create_step(
            workflow_id=workflow.workflow_id,
            kind="GENERATE_PLAN",
            idempotency_key="plan-attempt-1",
            input_data={
                "api_key": "must-not-be-stored",
                "prompt": "raw model prompt",
                "goal": "coverage",
            },
        )
        queued = await first.get_workflow(workflow.workflow_id)
        assert queued is not None
        assert queued.steps[0].input_data["api_key"] == "***"
        assert queued.steps[0].input_data["prompt"] == "[omitted from workflow metadata]"
        await first.start_step(step.step_id)

        restarted = SQLiteWorkflowStore(database)
        await restarted.initialize()
        interrupted = await restarted.get_workflow(workflow.workflow_id)
        assert interrupted is not None
        assert interrupted.status == "INTERRUPTED"
        assert interrupted.error is not None
        assert interrupted.error.code == "WORKFLOW_PROCESS_RESTART"
        assert interrupted.steps[0].status == "INTERRUPTED"

        resumed = await restarted.resume_workflow(workflow.workflow_id)
        assert resumed.status == "READY"
        assert resumed.error is None
        assert [event.revision for event in await restarted.list_events(workflow.workflow_id)] == [
            1,
            2,
            3,
            4,
            5,
        ]

    asyncio.run(exercise())


def test_orchestrator_bounds_ai_steps_and_fails_closed(tmp_path: Path) -> None:
    async def exercise() -> None:
        store = SQLiteWorkflowStore(tmp_path / "orchestrator.sqlite3")
        await store.initialize()
        await store.create_project("campus", "Campus Ops")
        first = await store.create_workflow(
            project_id="campus",
            name="First generation",
            description="",
            idempotency_key="first",
        )
        second = await store.create_workflow(
            project_id="campus",
            name="Second generation",
            description="",
            idempotency_key="second",
        )
        failed = await store.create_workflow(
            project_id="campus",
            name="Failed generation",
            description="",
            idempotency_key="failed",
        )
        orchestrator = WorkflowOrchestrator(store, max_concurrent_ai_steps=1)
        active = 0
        maximum = 0

        async def handler(step):  # type: ignore[no-untyped-def]
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            return []

        first_step = await orchestrator.submit(
            workflow_id=first.workflow_id,
            kind="GENERATE_PLAN",
            idempotency_key="plan-1",
            input_data={"goal": "first"},
            handler=handler,
        )
        duplicate = await orchestrator.submit(
            workflow_id=first.workflow_id,
            kind="GENERATE_PLAN",
            idempotency_key="plan-1",
            input_data={"goal": "first"},
            handler=handler,
        )
        assert duplicate.step_id == first_step.step_id
        second_step = await orchestrator.submit(
            workflow_id=second.workflow_id,
            kind="GENERATE_PLAN",
            idempotency_key="plan-2",
            input_data={"goal": "second"},
            handler=handler,
        )
        await asyncio.gather(
            orchestrator.wait(first_step.step_id),
            orchestrator.wait(second_step.step_id),
        )
        assert maximum == 1
        assert (await store.get_workflow(first.workflow_id)).status == "READY"  # type: ignore[union-attr]
        assert (await store.get_workflow(second.workflow_id)).status == "READY"  # type: ignore[union-attr]

        async def broken_handler(step):  # type: ignore[no-untyped-def]
            del step
            raise RuntimeError("sensitive implementation detail")

        failed_step = await orchestrator.submit(
            workflow_id=failed.workflow_id,
            kind="GENERATE_TESTS",
            idempotency_key="tests-1",
            input_data={},
            handler=broken_handler,
        )
        ended = await orchestrator.wait(failed_step.step_id)
        assert ended is not None
        assert ended.status == "FAILED_SAFE"
        assert ended.error is not None
        assert ended.error.message == "workflow step raised RuntimeError"
        failed_workflow = await store.get_workflow(failed.workflow_id)
        assert failed_workflow is not None
        assert failed_workflow.status == "FAILED_SAFE"
        await orchestrator.shutdown()

    asyncio.run(exercise())


def test_verified_artifacts_and_approvals_survive_a_resumable_checkpoint(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        database = tmp_path / "checkpoint.sqlite3"
        store = SQLiteWorkflowStore(database)
        await store.initialize()
        await store.create_project(
            "campus",
            "Campus Ops",
            workspace_path="examples/campus_agent",
            default_spec_refs={"manifest": "agentqa.manifest.yaml"},
        )
        workflow = await store.create_workflow(
            project_id="campus",
            name="Verified checkpoint",
            description="",
            idempotency_key="checkpoint",
        )
        completed = await store.create_step(
            workflow_id=workflow.workflow_id,
            kind="GENERATE_PLAN",
            idempotency_key="plan-1",
            input_data={"goal": "verify"},
        )
        await store.start_step(completed.step_id)
        artifact = ArtifactRef(
            artifact_id="plan-json",
            artifact_type="TestPlan",
            uri=".agentqa/workflows/plan.json",
            schema_version="agentqa.test-plan/v1alpha1",
            checksum=f"sha256:{'0' * 64}",
            source="generator",
        )
        await store.finish_step(
            completed.step_id,
            artifacts=[artifact],
            complete_workflow=False,
        )
        approval = ApprovalDecision(
            approval_id="approve-plan-1",
            workflow_id=workflow.workflow_id,
            subject_kind="TestPlan",
            subject_id=artifact.artifact_id,
            subject_hash=artifact.checksum,
            decision="APPROVED",
            reviewer="local-user",
            comment="Reviewed in the workbench.",
            created_at=utc_now(),
        )
        await store.save_approval(approval)

        interrupted = await store.create_step(
            workflow_id=workflow.workflow_id,
            kind="GENERATE_TESTS",
            idempotency_key="tests-1",
            input_data={"plan_hash": artifact.checksum},
        )
        await store.start_step(interrupted.step_id)

        restarted = SQLiteWorkflowStore(database)
        await restarted.initialize()
        resumed = await restarted.resume_workflow(workflow.workflow_id)
        assert resumed.status == "READY"
        assert resumed.steps[0].artifacts == [artifact]
        assert await restarted.list_approvals(workflow.workflow_id) == [approval]
        project = await restarted.get_project("campus")
        assert project is not None
        assert project.default_spec_refs == {"manifest": "agentqa.manifest.yaml"}

    asyncio.run(exercise())
