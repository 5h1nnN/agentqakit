from pathlib import Path

import pytest

from agentqa.models import ErrorInfo
from agentqa.workbench.store import SQLiteWorkflowStore, WorkflowStoreError


@pytest.mark.asyncio
async def test_failed_workflow_allows_only_non_authoritative_diagnosis(tmp_path: Path) -> None:
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    await store.initialize()
    await store.create_project("project", "Project")
    workflow = await store.create_workflow(
        project_id="project", name="Evaluation", description="", idempotency_key="evaluation"
    )
    failed_step = await store.create_step(
        workflow_id=workflow.workflow_id,
        kind="PREFLIGHT",
        idempotency_key="preflight",
        input_data={},
    )
    await store.start_step(failed_step.step_id)
    original_error = ErrorInfo(code="ARTIFACT_CATALOG_CHECKSUM_ERROR", message="checksum mismatch")
    await store.fail_step(failed_step.step_id, original_error)

    with pytest.raises(WorkflowStoreError, match="workflow must be READY"):
        await store.create_step(
            workflow_id=workflow.workflow_id,
            kind="RUN_CAMPAIGN",
            idempotency_key="retry",
            input_data={},
        )

    diagnostic = await store.create_step(
        workflow_id=workflow.workflow_id,
        kind="ANALYZE_FAILURE",
        idempotency_key="diagnose",
        input_data={"diagnostic_only": True, "failure_id": "failure"},
    )
    await store.start_step(diagnostic.step_id)
    running = await store.get_workflow(workflow.workflow_id)
    assert running is not None and running.status == "FAILED_SAFE"
    assert running.error == original_error

    await store.finish_step(diagnostic.step_id, artifacts=[], complete_workflow=False)
    finished = await store.get_workflow(workflow.workflow_id)
    assert finished is not None and finished.status == "FAILED_SAFE"
    assert finished.error == original_error
    diagnostic_step = next(step for step in finished.steps if step.step_id == diagnostic.step_id)
    assert diagnostic_step.status == "COMPLETED"
