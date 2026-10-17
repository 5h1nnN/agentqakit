"""Transactional SQLite persistence for local-workbench workflows."""

from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
from collections.abc import Collection
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import JsonValue, ValidationError

from agentqa.conventions import new_id, to_utc_iso8601, utc_now
from agentqa.errors import AgentQAError
from agentqa.models import ErrorInfo
from agentqa.models.manifest import RunManifestSnapshot
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.storage.redaction import redact_secrets, redact_text_secrets
from agentqa.workbench.models import (
    ApprovalDecision,
    ArtifactRef,
    WorkbenchProject,
    WorkflowEvent,
    WorkflowRun,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepKind,
    WorkflowStepStatus,
)

WORKFLOW_STORE_SCHEMA_VERSION = "agentqa.workflow-store/v1alpha1"
_MAX_SUMMARY_BYTES = 64 * 1024
_OMITTED_CONTENT_KEYS = frozenset(
    {"prompt", "rawprompt", "messages", "rawoutput", "modeloutput", "completion", "response"}
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workbench_projects (
    project_id TEXT PRIMARY KEY,
    project_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workflow_runs (
    workflow_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES workbench_projects(project_id),
    status TEXT NOT NULL,
    revision INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    workflow_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_workflows_project_updated
    ON workflow_runs(project_id, updated_at DESC, workflow_id DESC);
CREATE TABLE IF NOT EXISTS workflow_steps (
    step_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(workflow_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    step_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(workflow_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_workflow_steps_order
    ON workflow_steps(workflow_id, created_at, step_id);
CREATE TABLE IF NOT EXISTS workflow_artifacts (
    artifact_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(workflow_id) ON DELETE CASCADE,
    step_id TEXT NOT NULL REFERENCES workflow_steps(step_id) ON DELETE CASCADE,
    artifact_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(workflow_id, artifact_id)
);
CREATE INDEX IF NOT EXISTS idx_workflow_artifacts
    ON workflow_artifacts(workflow_id, step_id, artifact_id);
CREATE TABLE IF NOT EXISTS workflow_events (
    event_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(workflow_id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    UNIQUE(workflow_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_workflow_events_order
    ON workflow_events(workflow_id, revision);
CREATE TABLE IF NOT EXISTS workflow_approvals (
    approval_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(workflow_id) ON DELETE CASCADE,
    subject_hash TEXT NOT NULL,
    approval_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workflow_approvals
    ON workflow_approvals(workflow_id, created_at, approval_id);
"""

_ALLOWED_TRANSITIONS: dict[WorkflowStatus, frozenset[WorkflowStatus]] = {
    "DRAFT": frozenset({"READY", "CANCELLED"}),
    "READY": frozenset({"RUNNING", "WAITING_EXTERNAL_REPAIR", "CANCELLED"}),
    "RUNNING": frozenset(
        {
            "READY",
            "WAITING_REVIEW",
            "WAITING_EXTERNAL_REPAIR",
            "INTERRUPTED",
            "FAILED_SAFE",
            "COMPLETED",
            "CANCELLED",
        }
    ),
    "WAITING_REVIEW": frozenset({"RUNNING", "CANCELLED"}),
    "WAITING_EXTERNAL_REPAIR": frozenset({"READY", "CANCELLED"}),
    "INTERRUPTED": frozenset({"READY", "CANCELLED"}),
    "FAILED_SAFE": frozenset(),
    "COMPLETED": frozenset(),
    "CANCELLED": frozenset(),
}


class WorkflowStoreError(AgentQAError):
    """Expected workbench persistence or state-transition error."""

    def __init__(self, message: str, *, code: str = "WORKFLOW_STORE_ERROR") -> None:
        super().__init__(message, code=code)


class WorkflowStore(Protocol):
    """Persistence boundary used by the workbench API and orchestrator."""

    async def initialize(self) -> None: ...

    async def create_project(
        self,
        project_id: str,
        name: str,
        description: str = "",
        *,
        workspace_path: str = ".",
        default_spec_refs: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> WorkbenchProject: ...

    async def get_project(self, project_id: str) -> WorkbenchProject | None: ...

    async def list_projects(self) -> list[WorkbenchProject]: ...

    async def update_project_observation(
        self, project_id: str, manifest_snapshot: RunManifestSnapshot
    ) -> WorkbenchProject: ...

    async def update_project_spec_refs(
        self, project_id: str, default_spec_refs: dict[str, str]
    ) -> WorkbenchProject: ...

    async def create_workflow(
        self,
        *,
        project_id: str,
        name: str,
        description: str,
        idempotency_key: str,
        draft: bool = False,
    ) -> WorkflowRun: ...

    async def get_workflow(self, workflow_id: str) -> WorkflowRun | None: ...

    async def list_workflows(self, project_id: str | None = None) -> list[WorkflowRun]: ...

    async def list_events(
        self, workflow_id: str, *, after_revision: int = 0
    ) -> list[WorkflowEvent]: ...

    async def create_step(
        self,
        *,
        workflow_id: str,
        kind: WorkflowStepKind,
        idempotency_key: str,
        input_data: dict[str, JsonValue],
    ) -> WorkflowStep: ...

    async def start_step(self, step_id: str) -> WorkflowStep: ...

    async def finish_step(
        self,
        step_id: str,
        *,
        artifacts: list[ArtifactRef],
        complete_workflow: bool,
    ) -> WorkflowStep: ...

    async def fail_step(self, step_id: str, error: ErrorInfo) -> WorkflowStep: ...

    async def interrupt_step(self, step_id: str, error: ErrorInfo) -> WorkflowStep: ...

    async def cancel_workflow(self, workflow_id: str) -> WorkflowRun: ...

    async def resume_workflow(self, workflow_id: str) -> WorkflowRun: ...

    async def wait_for_external_repair(
        self, workflow_id: str, *, handoff_id: str, checksum: str
    ) -> WorkflowRun: ...

    async def save_approval(self, decision: ApprovalDecision) -> ApprovalDecision: ...

    async def list_approvals(self, workflow_id: str) -> list[ApprovalDecision]: ...

    async def record_event(
        self,
        workflow_id: str,
        *,
        event_type: str,
        message: str,
        data: dict[str, JsonValue] | None = None,
    ) -> WorkflowEvent: ...

    async def close(self) -> None: ...


def _json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _parse(model: type[Any], payload: str, label: str) -> Any:
    try:
        return model.model_validate_json(payload)
    except ValidationError:
        raise WorkflowStoreError(
            f"stored {label} does not match the expected schema",
            code="WORKFLOW_STORE_FORMAT_ERROR",
        ) from None


def _safe_summary(value: dict[str, JsonValue], label: str) -> dict[str, JsonValue]:
    """Redact secrets and raw model content from bounded persisted summaries."""

    def visit(item: Any, key: str | None = None) -> Any:
        normalized = "" if key is None else re.sub(r"[^a-z0-9]", "", key.lower())
        if normalized in _OMITTED_CONTENT_KEYS:
            return "[omitted from workflow metadata]"
        if isinstance(item, dict):
            return {
                str(child_key): visit(child, str(child_key)) for child_key, child in item.items()
            }
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, str):
            return redact_text_secrets(item)
        return item

    safe = cast(dict[str, JsonValue], visit(redact_secrets(value)))
    if len(canonical_json_bytes(safe)) > _MAX_SUMMARY_BYTES:
        raise WorkflowStoreError(
            f"{label} exceeds the workflow metadata limit",
            code="WORKFLOW_SUMMARY_TOO_LARGE",
        )
    return safe


class SQLiteWorkflowStore:
    """Persist resumable workflow metadata beside the existing result store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._init_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @staticmethod
    def _parse_project(payload: str) -> WorkbenchProject:
        return cast(WorkbenchProject, _parse(WorkbenchProject, payload, "project"))

    @staticmethod
    def _parse_step(payload: str) -> WorkflowStep:
        return cast(WorkflowStep, _parse(WorkflowStep, payload, "workflow step"))

    @staticmethod
    def _parse_event(payload: str) -> WorkflowEvent:
        return cast(WorkflowEvent, _parse(WorkflowEvent, payload, "workflow event"))

    @staticmethod
    def _parse_artifact(payload: str) -> ArtifactRef:
        return cast(ArtifactRef, _parse(ArtifactRef, payload, "workflow artifact"))

    @staticmethod
    def _parse_approval(payload: str) -> ApprovalDecision:
        return cast(ApprovalDecision, _parse(ApprovalDecision, payload, "workflow approval"))

    def _workflow_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> WorkflowRun:
        workflow = cast(
            WorkflowRun,
            _parse(WorkflowRun, cast(str, row["workflow_json"]), "workflow"),
        )
        step_rows = connection.execute(
            "SELECT step_json FROM workflow_steps WHERE workflow_id = ? "
            "ORDER BY created_at, step_id",
            (workflow.workflow_id,),
        ).fetchall()
        value = workflow.model_dump(mode="python")
        value["steps"] = [
            self._parse_step(cast(str, step_row["step_json"])).model_dump(mode="python")
            for step_row in step_rows
        ]
        try:
            return WorkflowRun.model_validate(value)
        except ValidationError:
            raise WorkflowStoreError(
                "stored workflow aggregate does not match the expected schema",
                code="WORKFLOW_STORE_FORMAT_ERROR",
            ) from None

    @staticmethod
    def _write_workflow(connection: sqlite3.Connection, workflow: WorkflowRun) -> None:
        value = workflow.model_copy(update={"steps": []})
        connection.execute(
            """UPDATE workflow_runs
               SET status = ?, revision = ?, workflow_json = ?, updated_at = ?
               WHERE workflow_id = ?""",
            (
                workflow.status,
                workflow.revision,
                _json(value.model_dump(mode="json")),
                to_utc_iso8601(workflow.updated_at),
                workflow.workflow_id,
            ),
        )

    @staticmethod
    def _write_step(connection: sqlite3.Connection, step: WorkflowStep) -> None:
        connection.execute(
            "UPDATE workflow_steps SET status = ?, step_json = ? WHERE step_id = ?",
            (step.status, _json(step.model_dump(mode="json")), step.step_id),
        )

    def _write_artifacts(
        self,
        connection: sqlite3.Connection,
        workflow_id: str,
        step_id: str,
        artifacts: list[ArtifactRef],
    ) -> None:
        for artifact in artifacts:
            row = connection.execute(
                "SELECT artifact_json, step_id FROM workflow_artifacts "
                "WHERE workflow_id = ? AND artifact_id = ?",
                (workflow_id, artifact.artifact_id),
            ).fetchone()
            if row is not None:
                if (
                    self._parse_artifact(cast(str, row["artifact_json"])) != artifact
                    or row["step_id"] != step_id
                ):
                    raise WorkflowStoreError(
                        "artifact_id already exists with different evidence",
                        code="WORKFLOW_ARTIFACT_CONFLICT",
                    )
                continue
            connection.execute(
                """INSERT INTO workflow_artifacts(
                       artifact_id, workflow_id, step_id, artifact_json, created_at
                   ) VALUES(?, ?, ?, ?, ?)""",
                (
                    artifact.artifact_id,
                    workflow_id,
                    step_id,
                    _json(artifact.model_dump(mode="json")),
                    to_utc_iso8601(utc_now()),
                ),
            )

    def _append_event(
        self,
        connection: sqlite3.Connection,
        workflow: WorkflowRun,
        *,
        event_type: str,
        message: str,
        step_id: str | None = None,
        data: dict[str, JsonValue] | None = None,
    ) -> tuple[WorkflowRun, WorkflowEvent]:
        now = utc_now()
        revision = workflow.revision + 1
        event = WorkflowEvent(
            event_id=new_id(),
            workflow_id=workflow.workflow_id,
            revision=revision,
            event_type=event_type,
            occurred_at=now,
            message=message,
            step_id=step_id,
            data={} if data is None else data,
        )
        updated = workflow.model_copy(update={"revision": revision, "updated_at": now})
        connection.execute(
            """INSERT INTO workflow_events(
                   event_id, workflow_id, revision, event_json, occurred_at
               ) VALUES(?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.workflow_id,
                event.revision,
                _json(event.model_dump(mode="json")),
                to_utc_iso8601(event.occurred_at),
            ),
        )
        self._write_workflow(connection, updated)
        return updated, event

    def _load_workflow(self, connection: sqlite3.Connection, workflow_id: str) -> WorkflowRun:
        row = connection.execute(
            "SELECT workflow_json FROM workflow_runs WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            raise WorkflowStoreError("workflow does not exist", code="WORKFLOW_NOT_FOUND")
        return self._workflow_from_row(connection, row)

    def _load_step(self, connection: sqlite3.Connection, step_id: str) -> WorkflowStep:
        row = connection.execute(
            "SELECT step_json FROM workflow_steps WHERE step_id = ?", (step_id,)
        ).fetchone()
        if row is None:
            raise WorkflowStoreError("workflow step does not exist", code="WORKFLOW_STEP_NOT_FOUND")
        return self._parse_step(cast(str, row["step_json"]))

    def _transition(
        self,
        connection: sqlite3.Connection,
        workflow: WorkflowRun,
        status: WorkflowStatus,
        *,
        event_type: str,
        message: str,
        step_id: str | None = None,
        error: ErrorInfo | None = None,
    ) -> WorkflowRun:
        if status not in _ALLOWED_TRANSITIONS[workflow.status]:
            raise WorkflowStoreError(
                f"workflow cannot transition from {workflow.status} to {status}",
                code="WORKFLOW_TRANSITION_ERROR",
            )
        now = utc_now()
        updates: dict[str, Any] = {"status": status, "updated_at": now, "error": error}
        if status == "RUNNING" and workflow.started_at is None:
            updates["started_at"] = now
        if status in {"COMPLETED", "FAILED_SAFE", "CANCELLED"}:
            updates["finished_at"] = now
        updated = workflow.model_copy(update=updates)
        updated, _ = self._append_event(
            connection,
            updated,
            event_type=event_type,
            message=message,
            step_id=step_id,
            data={"from_status": workflow.status, "to_status": status},
        )
        return updated

    def _initialize_sync(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(_SCHEMA)
                row = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key = 'workflow_schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO schema_metadata(key, value) VALUES(?, ?)",
                        ("workflow_schema_version", WORKFLOW_STORE_SCHEMA_VERSION),
                    )
                elif row["value"] != WORKFLOW_STORE_SCHEMA_VERSION:
                    raise WorkflowStoreError(
                        "SQLite workflow-store schema is not supported",
                        code="WORKFLOW_STORE_SCHEMA_ERROR",
                    )
                self._interrupt_running(connection)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise WorkflowStoreError(
                    f"SQLite workflow initialization failed: {type(exc).__name__}",
                    code="WORKFLOW_STORE_DATABASE_ERROR",
                ) from None
            finally:
                connection.close()
            self._initialized = True

    def _interrupt_running(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT workflow_json FROM workflow_runs WHERE status = 'RUNNING'"
        ).fetchall()
        for row in rows:
            workflow = self._workflow_from_row(connection, row)
            error = ErrorInfo(
                code="WORKFLOW_PROCESS_RESTART",
                message="the local process stopped before the workflow step completed",
            )
            now = utc_now()
            step_rows = connection.execute(
                "SELECT step_json FROM workflow_steps WHERE workflow_id = ? AND status = 'RUNNING'",
                (workflow.workflow_id,),
            ).fetchall()
            for step_row in step_rows:
                step = self._parse_step(cast(str, step_row["step_json"]))
                interrupted = step.model_copy(
                    update={"status": "INTERRUPTED", "finished_at": now, "error": error}
                )
                self._write_step(connection, interrupted)
            interrupted_workflow = workflow.model_copy(
                update={"status": "INTERRUPTED", "updated_at": now, "error": error}
            )
            self._append_event(
                connection,
                interrupted_workflow,
                event_type="WORKFLOW_INTERRUPTED",
                message="Workflow marked interrupted after process restart.",
                data={"reason": "process_restart"},
            )

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _create_project_sync(
        self,
        project_id: str,
        name: str,
        description: str,
        workspace_path: str,
        default_spec_refs: dict[str, str],
        tags: list[str],
    ) -> WorkbenchProject:
        self._initialize_sync()
        now = utc_now()
        project = WorkbenchProject(
            project_id=project_id,
            name=redact_text_secrets(name),
            description=redact_text_secrets(description),
            workspace_path=workspace_path,
            default_spec_refs=default_spec_refs,
            tags=tags,
            created_at=now,
            updated_at=now,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT project_json FROM workbench_projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if row is not None:
                existing = self._parse_project(cast(str, row["project_json"]))
                if (
                    existing.name != project.name
                    or existing.description != project.description
                    or existing.workspace_path != project.workspace_path
                    or existing.default_spec_refs != project.default_spec_refs
                    or existing.tags != project.tags
                ):
                    raise WorkflowStoreError(
                        "project_id already exists with different configuration",
                        code="WORKBENCH_PROJECT_CONFLICT",
                    )
                connection.rollback()
                return existing
            connection.execute(
                """INSERT INTO workbench_projects(
                       project_id, project_json, created_at, updated_at
                   ) VALUES(?, ?, ?, ?)""",
                (
                    project.project_id,
                    _json(project.model_dump(mode="json")),
                    to_utc_iso8601(now),
                    to_utc_iso8601(now),
                ),
            )
            connection.commit()
            return project
        except WorkflowStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise WorkflowStoreError(
                f"SQLite project transaction failed: {type(exc).__name__}",
                code="WORKFLOW_STORE_DATABASE_ERROR",
            ) from None
        finally:
            connection.close()

    async def create_project(
        self,
        project_id: str,
        name: str,
        description: str = "",
        *,
        workspace_path: str = ".",
        default_spec_refs: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> WorkbenchProject:
        return await asyncio.to_thread(
            self._create_project_sync,
            project_id,
            name,
            description,
            workspace_path,
            {} if default_spec_refs is None else default_spec_refs,
            [] if tags is None else tags,
        )

    def _get_project_sync(self, project_id: str) -> WorkbenchProject | None:
        self._initialize_sync()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT project_json FROM workbench_projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            return None if row is None else self._parse_project(cast(str, row["project_json"]))
        finally:
            connection.close()

    async def get_project(self, project_id: str) -> WorkbenchProject | None:
        return await asyncio.to_thread(self._get_project_sync, project_id)

    def _list_projects_sync(self) -> list[WorkbenchProject]:
        self._initialize_sync()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT project_json FROM workbench_projects ORDER BY updated_at DESC, project_id"
            ).fetchall()
            return [self._parse_project(cast(str, row["project_json"])) for row in rows]
        finally:
            connection.close()

    async def list_projects(self) -> list[WorkbenchProject]:
        return await asyncio.to_thread(self._list_projects_sync)

    def _update_project_observation_sync(
        self, project_id: str, manifest_snapshot: RunManifestSnapshot
    ) -> WorkbenchProject:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT project_json FROM workbench_projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if row is None:
                raise WorkflowStoreError(
                    "workbench project does not exist", code="WORKBENCH_PROJECT_NOT_FOUND"
                )
            project = self._parse_project(cast(str, row["project_json"]))
            updated = project.model_copy(
                update={"last_manifest_snapshot": manifest_snapshot, "updated_at": utc_now()}
            )
            connection.execute(
                "UPDATE workbench_projects SET project_json = ?, updated_at = ? "
                "WHERE project_id = ?",
                (
                    _json(updated.model_dump(mode="json")),
                    to_utc_iso8601(updated.updated_at),
                    project_id,
                ),
            )
            connection.commit()
            return updated
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def update_project_observation(
        self, project_id: str, manifest_snapshot: RunManifestSnapshot
    ) -> WorkbenchProject:
        return await asyncio.to_thread(
            self._update_project_observation_sync, project_id, manifest_snapshot
        )

    def _update_project_spec_refs_sync(
        self, project_id: str, default_spec_refs: dict[str, str]
    ) -> WorkbenchProject:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT project_json FROM workbench_projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if row is None:
                raise WorkflowStoreError(
                    "workbench project does not exist", code="WORKBENCH_PROJECT_NOT_FOUND"
                )
            project = self._parse_project(cast(str, row["project_json"]))
            updated = project.model_copy(
                update={"default_spec_refs": default_spec_refs, "updated_at": utc_now()}
            )
            connection.execute(
                "UPDATE workbench_projects SET project_json = ?, updated_at = ? "
                "WHERE project_id = ?",
                (
                    _json(updated.model_dump(mode="json")),
                    to_utc_iso8601(updated.updated_at),
                    project_id,
                ),
            )
            connection.commit()
            return updated
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def update_project_spec_refs(
        self, project_id: str, default_spec_refs: dict[str, str]
    ) -> WorkbenchProject:
        return await asyncio.to_thread(
            self._update_project_spec_refs_sync, project_id, default_spec_refs
        )

    def _create_workflow_sync(
        self,
        project_id: str,
        name: str,
        description: str,
        idempotency_key: str,
        draft: bool,
    ) -> WorkflowRun:
        self._initialize_sync()
        material: dict[str, JsonValue] = {
            "project_id": project_id,
            "name": redact_text_secrets(name),
            "description": redact_text_secrets(description),
            "draft": draft,
        }
        safe_material = _safe_summary(material, "workflow input")
        input_hash = sha256_digest(canonical_json_bytes(safe_material))
        now = utc_now()
        workflow = WorkflowRun(
            workflow_id=new_id(),
            project_id=project_id,
            name=cast(str, safe_material["name"]),
            description=cast(str, safe_material["description"]),
            status="DRAFT" if draft else "READY",
            revision=1,
            idempotency_key=idempotency_key,
            input_hash=input_hash,
            input_snapshot=safe_material,
            created_at=now,
            updated_at=now,
        )
        event = WorkflowEvent(
            event_id=new_id(),
            workflow_id=workflow.workflow_id,
            revision=1,
            event_type="WORKFLOW_CREATED",
            occurred_at=now,
            message="Workflow draft created." if draft else "Workflow is ready.",
            data={"status": workflow.status},
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    "SELECT 1 FROM workbench_projects WHERE project_id = ?", (project_id,)
                ).fetchone()
                is None
            ):
                raise WorkflowStoreError(
                    "workbench project does not exist", code="WORKBENCH_PROJECT_NOT_FOUND"
                )
            row = connection.execute(
                "SELECT workflow_json FROM workflow_runs "
                "WHERE project_id = ? AND idempotency_key = ?",
                (project_id, idempotency_key),
            ).fetchone()
            if row is not None:
                existing = self._workflow_from_row(connection, row)
                if existing.input_hash != input_hash:
                    raise WorkflowStoreError(
                        "idempotency key already exists with different workflow input",
                        code="WORKFLOW_IDEMPOTENCY_CONFLICT",
                    )
                connection.rollback()
                return existing
            connection.execute(
                """INSERT INTO workflow_runs(
                       workflow_id, project_id, status, revision, idempotency_key,
                       input_hash, workflow_json, created_at, updated_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    workflow.workflow_id,
                    workflow.project_id,
                    workflow.status,
                    workflow.revision,
                    workflow.idempotency_key,
                    workflow.input_hash,
                    _json(workflow.model_dump(mode="json")),
                    to_utc_iso8601(now),
                    to_utc_iso8601(now),
                ),
            )
            connection.execute(
                """INSERT INTO workflow_events(
                       event_id, workflow_id, revision, event_json, occurred_at
                   ) VALUES(?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.workflow_id,
                    event.revision,
                    _json(event.model_dump(mode="json")),
                    to_utc_iso8601(now),
                ),
            )
            project_row = connection.execute(
                "SELECT project_json FROM workbench_projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if project_row is None:
                raise WorkflowStoreError(
                    "workbench project does not exist", code="WORKBENCH_PROJECT_NOT_FOUND"
                )
            project = self._parse_project(cast(str, project_row["project_json"]))
            updated_project = project.model_copy(
                update={"recent_workflow_id": workflow.workflow_id, "updated_at": now}
            )
            connection.execute(
                "UPDATE workbench_projects SET project_json = ?, updated_at = ? "
                "WHERE project_id = ?",
                (
                    _json(updated_project.model_dump(mode="json")),
                    to_utc_iso8601(now),
                    project_id,
                ),
            )
            connection.commit()
            return workflow
        except WorkflowStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise WorkflowStoreError(
                f"SQLite workflow transaction failed: {type(exc).__name__}",
                code="WORKFLOW_STORE_DATABASE_ERROR",
            ) from None
        finally:
            connection.close()

    async def create_workflow(
        self,
        *,
        project_id: str,
        name: str,
        description: str,
        idempotency_key: str,
        draft: bool = False,
    ) -> WorkflowRun:
        return await asyncio.to_thread(
            self._create_workflow_sync,
            project_id,
            name,
            description,
            idempotency_key,
            draft,
        )

    def _get_workflow_sync(self, workflow_id: str) -> WorkflowRun | None:
        self._initialize_sync()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT workflow_json FROM workflow_runs WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
            return None if row is None else self._workflow_from_row(connection, row)
        finally:
            connection.close()

    async def get_workflow(self, workflow_id: str) -> WorkflowRun | None:
        return await asyncio.to_thread(self._get_workflow_sync, workflow_id)

    def _list_workflows_sync(self, project_id: str | None) -> list[WorkflowRun]:
        self._initialize_sync()
        connection = self._connect()
        try:
            if project_id is None:
                rows = connection.execute(
                    "SELECT workflow_json FROM workflow_runs "
                    "ORDER BY updated_at DESC, workflow_id DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT workflow_json FROM workflow_runs WHERE project_id = ? "
                    "ORDER BY updated_at DESC, workflow_id DESC",
                    (project_id,),
                ).fetchall()
            return [self._workflow_from_row(connection, row) for row in rows]
        finally:
            connection.close()

    async def list_workflows(self, project_id: str | None = None) -> list[WorkflowRun]:
        return await asyncio.to_thread(self._list_workflows_sync, project_id)

    def _list_events_sync(self, workflow_id: str, after_revision: int) -> list[WorkflowEvent]:
        self._initialize_sync()
        if after_revision < 0:
            raise WorkflowStoreError(
                "after_revision must be non-negative", code="WORKFLOW_EVENT_CURSOR_ERROR"
            )
        connection = self._connect()
        try:
            if (
                connection.execute(
                    "SELECT 1 FROM workflow_runs WHERE workflow_id = ?", (workflow_id,)
                ).fetchone()
                is None
            ):
                raise WorkflowStoreError("workflow does not exist", code="WORKFLOW_NOT_FOUND")
            rows = connection.execute(
                "SELECT event_json FROM workflow_events "
                "WHERE workflow_id = ? AND revision > ? ORDER BY revision",
                (workflow_id, after_revision),
            ).fetchall()
            return [self._parse_event(cast(str, row["event_json"])) for row in rows]
        finally:
            connection.close()

    async def list_events(
        self, workflow_id: str, *, after_revision: int = 0
    ) -> list[WorkflowEvent]:
        return await asyncio.to_thread(self._list_events_sync, workflow_id, after_revision)

    def _create_step_sync(
        self,
        workflow_id: str,
        kind: WorkflowStepKind,
        idempotency_key: str,
        input_data: dict[str, JsonValue],
    ) -> WorkflowStep:
        self._initialize_sync()
        safe_input = _safe_summary(input_data, "workflow step input")
        input_hash = sha256_digest(canonical_json_bytes(safe_input))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            workflow = self._load_workflow(connection, workflow_id)
            row = connection.execute(
                "SELECT step_json FROM workflow_steps "
                "WHERE workflow_id = ? AND idempotency_key = ?",
                (workflow_id, idempotency_key),
            ).fetchone()
            if row is not None:
                existing = self._parse_step(cast(str, row["step_json"]))
                if existing.input_hash != input_hash or existing.kind != kind:
                    raise WorkflowStoreError(
                        "idempotency key already exists with different step input",
                        code="WORKFLOW_IDEMPOTENCY_CONFLICT",
                    )
                connection.rollback()
                return existing
            if workflow.status != "READY":
                raise WorkflowStoreError(
                    "workflow must be READY before a step is queued",
                    code="WORKFLOW_TRANSITION_ERROR",
                )
            attempt_row = connection.execute(
                "SELECT COALESCE(MAX(attempt), 0) AS value FROM workflow_steps "
                "WHERE workflow_id = ? AND kind = ?",
                (workflow_id, kind),
            ).fetchone()
            attempt = int(attempt_row["value"]) + 1
            step = WorkflowStep(
                step_id=new_id(),
                workflow_id=workflow_id,
                kind=kind,
                status="QUEUED",
                attempt=attempt,
                idempotency_key=idempotency_key,
                input_hash=input_hash,
                input_data=safe_input,
                created_at=utc_now(),
            )
            connection.execute(
                """INSERT INTO workflow_steps(
                       step_id, workflow_id, kind, status, attempt, idempotency_key,
                       input_hash, step_json, created_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    step.step_id,
                    step.workflow_id,
                    step.kind,
                    step.status,
                    step.attempt,
                    step.idempotency_key,
                    step.input_hash,
                    _json(step.model_dump(mode="json")),
                    to_utc_iso8601(step.created_at),
                ),
            )
            self._append_event(
                connection,
                workflow,
                event_type="STEP_QUEUED",
                message=f"Queued {kind} attempt {attempt}.",
                step_id=step.step_id,
                data={"kind": kind, "attempt": attempt, "input_hash": input_hash},
            )
            connection.commit()
            return step
        except WorkflowStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise WorkflowStoreError(
                f"SQLite workflow-step transaction failed: {type(exc).__name__}",
                code="WORKFLOW_STORE_DATABASE_ERROR",
            ) from None
        finally:
            connection.close()

    async def create_step(
        self,
        *,
        workflow_id: str,
        kind: WorkflowStepKind,
        idempotency_key: str,
        input_data: dict[str, JsonValue],
    ) -> WorkflowStep:
        return await asyncio.to_thread(
            self._create_step_sync,
            workflow_id,
            kind,
            idempotency_key,
            input_data,
        )

    def _start_step_sync(self, step_id: str) -> WorkflowStep:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            step = self._load_step(connection, step_id)
            if step.status != "QUEUED":
                raise WorkflowStoreError(
                    "only a queued workflow step can start", code="WORKFLOW_STEP_TRANSITION_ERROR"
                )
            workflow = self._load_workflow(connection, step.workflow_id)
            self._transition(
                connection,
                workflow,
                "RUNNING",
                event_type="STEP_STARTED",
                message=f"Started {step.kind} attempt {step.attempt}.",
                step_id=step.step_id,
            )
            started = step.model_copy(update={"status": "RUNNING", "started_at": utc_now()})
            self._write_step(connection, started)
            connection.commit()
            return started
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def start_step(self, step_id: str) -> WorkflowStep:
        return await asyncio.to_thread(self._start_step_sync, step_id)

    def _finish_step_sync(
        self, step_id: str, artifacts: list[ArtifactRef], complete_workflow: bool
    ) -> WorkflowStep:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            step = self._load_step(connection, step_id)
            if step.status != "RUNNING":
                raise WorkflowStoreError(
                    "only a running workflow step can finish",
                    code="WORKFLOW_STEP_TRANSITION_ERROR",
                )
            safe_artifacts = [
                artifact.model_copy(
                    update={
                        "uri": redact_text_secrets(artifact.uri),
                        "source": redact_text_secrets(artifact.source),
                    }
                )
                for artifact in artifacts
            ]
            completed = step.model_copy(
                update={
                    "status": "COMPLETED",
                    "finished_at": utc_now(),
                    "artifacts": safe_artifacts,
                }
            )
            self._write_step(connection, completed)
            self._write_artifacts(connection, step.workflow_id, step.step_id, safe_artifacts)
            workflow = self._load_workflow(connection, step.workflow_id)
            target: WorkflowStatus = "COMPLETED" if complete_workflow else "READY"
            self._transition(
                connection,
                workflow,
                target,
                event_type="STEP_COMPLETED",
                message=f"Completed {step.kind} attempt {step.attempt}.",
                step_id=step.step_id,
            )
            connection.commit()
            return completed
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def finish_step(
        self,
        step_id: str,
        *,
        artifacts: list[ArtifactRef],
        complete_workflow: bool,
    ) -> WorkflowStep:
        return await asyncio.to_thread(
            self._finish_step_sync, step_id, artifacts, complete_workflow
        )

    def _end_step_sync(
        self,
        step_id: str,
        error: ErrorInfo,
        step_status: WorkflowStepStatus,
        workflow_status: WorkflowStatus,
        event_type: str,
    ) -> WorkflowStep:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            step = self._load_step(connection, step_id)
            if step.status != "RUNNING":
                if step.status == step_status:
                    connection.rollback()
                    return step
                raise WorkflowStoreError(
                    "only a running workflow step can be ended",
                    code="WORKFLOW_STEP_TRANSITION_ERROR",
                )
            ended = step.model_copy(
                update={"status": step_status, "finished_at": utc_now(), "error": error}
            )
            self._write_step(connection, ended)
            workflow = self._load_workflow(connection, step.workflow_id)
            self._transition(
                connection,
                workflow,
                workflow_status,
                event_type=event_type,
                message=error.message,
                step_id=step.step_id,
                error=error,
            )
            connection.commit()
            return ended
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def fail_step(self, step_id: str, error: ErrorInfo) -> WorkflowStep:
        return await asyncio.to_thread(
            self._end_step_sync,
            step_id,
            error,
            "FAILED_SAFE",
            "FAILED_SAFE",
            "STEP_FAILED_SAFE",
        )

    async def interrupt_step(self, step_id: str, error: ErrorInfo) -> WorkflowStep:
        return await asyncio.to_thread(
            self._end_step_sync,
            step_id,
            error,
            "INTERRUPTED",
            "INTERRUPTED",
            "STEP_INTERRUPTED",
        )

    def _cancel_workflow_sync(self, workflow_id: str) -> WorkflowRun:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            workflow = self._load_workflow(connection, workflow_id)
            if workflow.status == "CANCELLED":
                connection.rollback()
                return workflow
            updated = self._transition(
                connection,
                workflow,
                "CANCELLED",
                event_type="WORKFLOW_CANCELLED",
                message="Workflow cancelled by the user.",
            )
            now = utc_now()
            rows = connection.execute(
                "SELECT step_json FROM workflow_steps WHERE workflow_id = ? "
                "AND status IN ('QUEUED', 'RUNNING')",
                (workflow_id,),
            ).fetchall()
            for row in rows:
                step = self._parse_step(cast(str, row["step_json"]))
                self._write_step(
                    connection,
                    step.model_copy(update={"status": "CANCELLED", "finished_at": now}),
                )
            connection.commit()
            return updated
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def cancel_workflow(self, workflow_id: str) -> WorkflowRun:
        return await asyncio.to_thread(self._cancel_workflow_sync, workflow_id)

    def _resume_workflow_sync(self, workflow_id: str) -> WorkflowRun:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            workflow = self._load_workflow(connection, workflow_id)
            if sha256_digest(canonical_json_bytes(workflow.input_snapshot)) != workflow.input_hash:
                raise WorkflowStoreError(
                    "workflow input snapshot no longer matches its checksum",
                    code="WORKFLOW_CHECKPOINT_INVALID",
                )
            for step in workflow.steps:
                if sha256_digest(canonical_json_bytes(step.input_data)) != step.input_hash:
                    raise WorkflowStoreError(
                        "workflow step input no longer matches its checksum",
                        code="WORKFLOW_CHECKPOINT_INVALID",
                    )
                if step.status != "COMPLETED":
                    continue
                rows = connection.execute(
                    "SELECT artifact_json FROM workflow_artifacts WHERE step_id = ? "
                    "ORDER BY artifact_id",
                    (step.step_id,),
                ).fetchall()
                persisted = [self._parse_artifact(cast(str, row["artifact_json"])) for row in rows]
                if sorted(persisted, key=lambda artifact: artifact.artifact_id) != sorted(
                    step.artifacts, key=lambda artifact: artifact.artifact_id
                ) or any(artifact.verification_status != "VERIFIED" for artifact in persisted):
                    raise WorkflowStoreError(
                        "completed workflow artifacts are not fully verified",
                        code="WORKFLOW_CHECKPOINT_INVALID",
                    )
            updated = self._transition(
                connection,
                workflow,
                "READY",
                event_type="WORKFLOW_RESUMED",
                message="Workflow is ready to continue from its last verified checkpoint.",
            )
            connection.commit()
            return updated
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def resume_workflow(self, workflow_id: str) -> WorkflowRun:
        return await asyncio.to_thread(self._resume_workflow_sync, workflow_id)

    def _wait_for_external_repair_sync(
        self, workflow_id: str, handoff_id: str, checksum: str
    ) -> WorkflowRun:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            workflow = self._load_workflow(connection, workflow_id)
            updated = self._transition(
                connection,
                workflow,
                "WAITING_EXTERNAL_REPAIR",
                event_type="EXTERNAL_REPAIR_REQUESTED",
                message=(
                    "A bounded handoff bundle is ready for external repair: "
                    f"{handoff_id} ({checksum})."
                ),
            )
            connection.commit()
            return updated
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def wait_for_external_repair(
        self, workflow_id: str, *, handoff_id: str, checksum: str
    ) -> WorkflowRun:
        return await asyncio.to_thread(
            self._wait_for_external_repair_sync, workflow_id, handoff_id, checksum
        )

    def _save_approval_sync(self, decision: ApprovalDecision) -> ApprovalDecision:
        self._initialize_sync()
        safe = ApprovalDecision.model_validate(redact_secrets(decision.model_dump(mode="python")))
        safe = safe.model_copy(update={"comment": redact_text_secrets(safe.comment)})
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    "SELECT 1 FROM workflow_runs WHERE workflow_id = ?", (safe.workflow_id,)
                ).fetchone()
                is None
            ):
                raise WorkflowStoreError("workflow does not exist", code="WORKFLOW_NOT_FOUND")
            row = connection.execute(
                "SELECT approval_json FROM workflow_approvals WHERE approval_id = ?",
                (safe.approval_id,),
            ).fetchone()
            if row is not None:
                existing = self._parse_approval(cast(str, row["approval_json"]))
                if existing != safe:
                    raise WorkflowStoreError(
                        "approval_id already exists with different evidence",
                        code="WORKFLOW_APPROVAL_CONFLICT",
                    )
                connection.rollback()
                return existing
            connection.execute(
                """INSERT INTO workflow_approvals(
                       approval_id, workflow_id, subject_hash, approval_json, created_at
                   ) VALUES(?, ?, ?, ?, ?)""",
                (
                    safe.approval_id,
                    safe.workflow_id,
                    safe.subject_hash,
                    _json(safe.model_dump(mode="json")),
                    to_utc_iso8601(safe.created_at),
                ),
            )
            workflow = self._load_workflow(connection, safe.workflow_id)
            self._append_event(
                connection,
                workflow,
                event_type="APPROVAL_RECORDED",
                message=f"Recorded {safe.decision} decision for {safe.subject_kind}.",
                data={
                    "approval_id": safe.approval_id,
                    "subject_id": safe.subject_id,
                    "subject_hash": safe.subject_hash,
                    "decision": safe.decision,
                },
            )
            connection.commit()
            return safe
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def save_approval(self, decision: ApprovalDecision) -> ApprovalDecision:
        return await asyncio.to_thread(self._save_approval_sync, decision)

    def _list_approvals_sync(self, workflow_id: str) -> list[ApprovalDecision]:
        self._initialize_sync()
        connection = self._connect()
        try:
            if (
                connection.execute(
                    "SELECT 1 FROM workflow_runs WHERE workflow_id = ?", (workflow_id,)
                ).fetchone()
                is None
            ):
                raise WorkflowStoreError("workflow does not exist", code="WORKFLOW_NOT_FOUND")
            rows = connection.execute(
                "SELECT approval_json FROM workflow_approvals WHERE workflow_id = ? "
                "ORDER BY created_at, approval_id",
                (workflow_id,),
            ).fetchall()
            return [self._parse_approval(cast(str, row["approval_json"])) for row in rows]
        finally:
            connection.close()

    async def list_approvals(self, workflow_id: str) -> list[ApprovalDecision]:
        return await asyncio.to_thread(self._list_approvals_sync, workflow_id)

    def _record_event_sync(
        self,
        workflow_id: str,
        event_type: str,
        message: str,
        data: dict[str, JsonValue],
    ) -> WorkflowEvent:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            workflow = self._load_workflow(connection, workflow_id)
            _, event = self._append_event(
                connection,
                workflow,
                event_type=redact_text_secrets(event_type),
                message=redact_text_secrets(message),
                data=_safe_summary(data, "workflow event"),
            )
            connection.commit()
            return event
        except WorkflowStoreError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def record_event(
        self,
        workflow_id: str,
        *,
        event_type: str,
        message: str,
        data: dict[str, JsonValue] | None = None,
    ) -> WorkflowEvent:
        return await asyncio.to_thread(
            self._record_event_sync,
            workflow_id,
            event_type,
            message,
            {} if data is None else data,
        )

    async def close(self) -> None:
        return None


def valid_workflow_transition(current: WorkflowStatus, target: WorkflowStatus) -> bool:
    """Return whether the public state machine permits one transition."""
    return target in _ALLOWED_TRANSITIONS[current]


def ensure_expected_status(workflow: WorkflowRun, expected: Collection[WorkflowStatus]) -> None:
    """Raise a stable error when a caller races with another workflow action."""
    if workflow.status not in expected:
        allowed = ", ".join(sorted(expected))
        raise WorkflowStoreError(
            f"workflow status {workflow.status} is not one of: {allowed}",
            code="WORKFLOW_TRANSITION_ERROR",
        )


__all__ = [
    "WORKFLOW_STORE_SCHEMA_VERSION",
    "SQLiteWorkflowStore",
    "WorkflowStore",
    "WorkflowStoreError",
    "ensure_expected_status",
    "valid_workflow_transition",
]
