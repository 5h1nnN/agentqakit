"""Persistence for Web generation sessions and immutable human revisions."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from agentqa.errors import AgentQAError
from agentqa.specs.hashing import canonical_json_bytes
from agentqa.workbench.models import GeneratedTestRevision, GenerationSession

GENERATION_WORKBENCH_SCHEMA_VERSION = "agentqa.generation-workbench/v1alpha1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generation_sessions (
    session_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflow_runs(workflow_id) ON DELETE CASCADE,
    idempotency_key TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    session_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(workflow_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_generation_sessions_workflow
    ON generation_sessions(workflow_id, created_at DESC, session_id DESC);
CREATE TABLE IF NOT EXISTS generated_test_revisions (
    revision_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES generation_sessions(session_id) ON DELETE CASCADE,
    case_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    spec_hash TEXT NOT NULL,
    revision_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, case_id, revision),
    UNIQUE(session_id, case_id, spec_hash)
);
CREATE INDEX IF NOT EXISTS idx_generated_revisions_case
    ON generated_test_revisions(session_id, case_id, revision);
"""

_ALLOWED_STATUS_TRANSITIONS = {
    "PLANNING": frozenset({"PLAN_READY", "FAILED_SAFE"}),
    "PLAN_READY": frozenset({"GENERATING", "FAILED_SAFE"}),
    "GENERATING": frozenset({"CANDIDATES_READY", "FAILED_SAFE"}),
    "CANDIDATES_READY": frozenset(),
    "FAILED_SAFE": frozenset(),
}


class GenerationWorkbenchError(AgentQAError):
    def __init__(self, message: str, *, code: str = "GENERATION_WORKBENCH_ERROR") -> None:
        super().__init__(message, code=code)


def _json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


class GenerationWorkbenchStore:
    """Small versioned store sharing the workbench SQLite database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize_sync(self) -> None:
        with self._lock:
            if self._initialized:
                return
            connection = self._connect()
            try:
                connection.executescript(_SCHEMA)
                row = connection.execute(
                    "SELECT value FROM schema_metadata "
                    "WHERE key = 'generation_workbench_schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO schema_metadata(key, value) VALUES(?, ?)",
                        (
                            "generation_workbench_schema_version",
                            GENERATION_WORKBENCH_SCHEMA_VERSION,
                        ),
                    )
                elif row["value"] != GENERATION_WORKBENCH_SCHEMA_VERSION:
                    raise GenerationWorkbenchError(
                        "generation workbench schema is not supported",
                        code="GENERATION_WORKBENCH_SCHEMA_ERROR",
                    )
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    @staticmethod
    def _parse_session(payload: str) -> GenerationSession:
        try:
            return GenerationSession.model_validate_json(payload)
        except ValidationError:
            raise GenerationWorkbenchError(
                "stored generation session is invalid",
                code="GENERATION_SESSION_FORMAT_ERROR",
            ) from None

    @staticmethod
    def _parse_revision(payload: str) -> GeneratedTestRevision:
        try:
            return GeneratedTestRevision.model_validate_json(payload)
        except ValidationError:
            raise GenerationWorkbenchError(
                "stored generated-test revision is invalid",
                code="GENERATION_REVISION_FORMAT_ERROR",
            ) from None

    def _create_session_sync(self, session: GenerationSession) -> GenerationSession:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT session_json FROM generation_sessions "
                "WHERE workflow_id = ? AND idempotency_key = ?",
                (session.workflow_id, session.idempotency_key),
            ).fetchone()
            if row is not None:
                existing = self._parse_session(cast(str, row["session_json"]))
                if existing.input_hash != session.input_hash:
                    raise GenerationWorkbenchError(
                        "idempotency key already exists with different generation input",
                        code="GENERATION_SESSION_IDEMPOTENCY_CONFLICT",
                    )
                connection.rollback()
                return existing
            connection.execute(
                """INSERT INTO generation_sessions(
                       session_id, workflow_id, idempotency_key, input_hash, status,
                       session_json, created_at, updated_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.session_id,
                    session.workflow_id,
                    session.idempotency_key,
                    session.input_hash,
                    session.status,
                    _json(session.model_dump(mode="json")),
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
            connection.commit()
            return session
        except GenerationWorkbenchError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def create_session(self, session: GenerationSession) -> GenerationSession:
        return await asyncio.to_thread(self._create_session_sync, session)

    def _get_session_sync(self, session_id: str) -> GenerationSession | None:
        self._initialize_sync()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT session_json FROM generation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return None if row is None else self._parse_session(cast(str, row["session_json"]))
        finally:
            connection.close()

    async def get_session(self, session_id: str) -> GenerationSession | None:
        return await asyncio.to_thread(self._get_session_sync, session_id)

    def _list_sessions_sync(self, workflow_id: str) -> list[GenerationSession]:
        self._initialize_sync()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT session_json FROM generation_sessions WHERE workflow_id = ? "
                "ORDER BY created_at DESC, session_id DESC",
                (workflow_id,),
            ).fetchall()
            return [self._parse_session(cast(str, row["session_json"])) for row in rows]
        finally:
            connection.close()

    async def list_sessions(self, workflow_id: str) -> list[GenerationSession]:
        return await asyncio.to_thread(self._list_sessions_sync, workflow_id)

    def _update_session_sync(self, session: GenerationSession) -> GenerationSession:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT session_json FROM generation_sessions WHERE session_id = ?",
                (session.session_id,),
            ).fetchone()
            if row is None:
                raise GenerationWorkbenchError(
                    "generation session does not exist",
                    code="GENERATION_SESSION_NOT_FOUND",
                )
            previous = self._parse_session(cast(str, row["session_json"]))
            if (
                previous.workflow_id != session.workflow_id
                or previous.project_id != session.project_id
                or previous.input_hash != session.input_hash
            ):
                raise GenerationWorkbenchError(
                    "generation session identity is immutable",
                    code="GENERATION_SESSION_IDENTITY_ERROR",
                )
            allowed = _ALLOWED_STATUS_TRANSITIONS[previous.status]
            if session.status != previous.status and session.status not in allowed:
                raise GenerationWorkbenchError(
                    "generation session cannot transition from "
                    f"{previous.status} to {session.status}",
                    code="GENERATION_SESSION_TRANSITION_ERROR",
                )
            connection.execute(
                "UPDATE generation_sessions SET status = ?, session_json = ?, updated_at = ? "
                "WHERE session_id = ?",
                (
                    session.status,
                    _json(session.model_dump(mode="json")),
                    session.updated_at.isoformat(),
                    session.session_id,
                ),
            )
            connection.commit()
            return session
        except GenerationWorkbenchError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def update_session(self, session: GenerationSession) -> GenerationSession:
        return await asyncio.to_thread(self._update_session_sync, session)

    def _save_revision_sync(self, revision: GeneratedTestRevision) -> GeneratedTestRevision:
        self._initialize_sync()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision_json FROM generated_test_revisions "
                "WHERE session_id = ? AND case_id = ? AND spec_hash = ?",
                (revision.session_id, revision.case_id, revision.spec_hash),
            ).fetchone()
            if row is not None:
                connection.rollback()
                return self._parse_revision(cast(str, row["revision_json"]))
            latest = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) AS value FROM generated_test_revisions "
                "WHERE session_id = ? AND case_id = ?",
                (revision.session_id, revision.case_id),
            ).fetchone()
            expected = int(latest["value"]) + 1
            if revision.revision != expected:
                raise GenerationWorkbenchError(
                    "generated-test revision number is stale",
                    code="GENERATION_REVISION_CONFLICT",
                )
            connection.execute(
                """INSERT INTO generated_test_revisions(
                       revision_id, session_id, case_id, revision, spec_hash,
                       revision_json, created_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (
                    revision.revision_id,
                    revision.session_id,
                    revision.case_id,
                    revision.revision,
                    revision.spec_hash,
                    _json(revision.model_dump(mode="json", by_alias=True)),
                    revision.created_at.isoformat(),
                ),
            )
            connection.commit()
            return revision
        except GenerationWorkbenchError:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def save_revision(self, revision: GeneratedTestRevision) -> GeneratedTestRevision:
        return await asyncio.to_thread(self._save_revision_sync, revision)

    def _list_revisions_sync(
        self, session_id: str, case_id: str | None
    ) -> list[GeneratedTestRevision]:
        self._initialize_sync()
        connection = self._connect()
        try:
            if case_id is None:
                rows = connection.execute(
                    "SELECT revision_json FROM generated_test_revisions WHERE session_id = ? "
                    "ORDER BY case_id, revision",
                    (session_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT revision_json FROM generated_test_revisions "
                    "WHERE session_id = ? AND case_id = ? ORDER BY revision",
                    (session_id, case_id),
                ).fetchall()
            return [self._parse_revision(cast(str, row["revision_json"])) for row in rows]
        finally:
            connection.close()

    async def list_revisions(
        self, session_id: str, case_id: str | None = None
    ) -> list[GeneratedTestRevision]:
        return await asyncio.to_thread(self._list_revisions_sync, session_id, case_id)

    async def close(self) -> None:
        return None


__all__ = [
    "GENERATION_WORKBENCH_SCHEMA_VERSION",
    "GenerationWorkbenchError",
    "GenerationWorkbenchStore",
]
