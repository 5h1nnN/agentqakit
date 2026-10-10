"""Transactional SQLite implementation of the ResultStore protocol."""

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from agentqa.conventions import to_utc_iso8601, utc_now
from agentqa.models import RunResult, TestResult, TraceRecord
from agentqa.models.storage import FindingReview, StoredFinding, StoredRun, StoredRunSummary
from agentqa.specs.hashing import canonical_json_bytes
from agentqa.storage.redaction import redact_secrets
from agentqa.storage.results import (
    RESULT_STORE_SCHEMA_VERSION,
    ResultStoreError,
    findings_for,
    read_trace_artifact,
    validate_project_id,
    validate_run_id,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    status TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    mutation_score_json TEXT,
    run_json TEXT NOT NULL,
    artifact_path TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_project_started
    ON runs(project_id, started_at DESC, run_id DESC);
CREATE TABLE IF NOT EXISTS test_results (
    trial_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    test_case_id TEXT NOT NULL,
    passed INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_test_results_run
    ON test_results(run_id, test_case_id, trial_id);
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    test_case_id TEXT NOT NULL,
    trial_id TEXT NOT NULL REFERENCES test_results(trial_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    artifact_root TEXT,
    trace_ref TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_run ON traces(run_id, trace_id);
CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    trial_id TEXT NOT NULL REFERENCES test_results(trial_id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    finding_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(run_id, finding_id);
CREATE TABLE IF NOT EXISTS review_status (
    finding_id TEXT PRIMARY KEY REFERENCES findings(finding_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    note TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    review_json TEXT NOT NULL
);
"""


def _json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


class SQLiteResultStore:
    """Index run evidence transactionally while retaining large files externally."""

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
                    "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO schema_metadata(key, value) VALUES('schema_version', ?)",
                        (RESULT_STORE_SCHEMA_VERSION,),
                    )
                elif row["value"] != RESULT_STORE_SCHEMA_VERSION:
                    raise ResultStoreError(
                        "SQLite result-store schema is not supported",
                        code="RESULT_STORE_SCHEMA_ERROR",
                    )
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    @staticmethod
    def _prepare_stored(
        run: RunResult,
        project_id: str,
        artifact_path: str | Path | None,
    ) -> StoredRun:
        resolved_artifact: str | None = None
        if artifact_path is not None:
            artifact = Path(artifact_path).resolve()
            if not artifact.is_dir() or artifact.name != run.run_id:
                raise ResultStoreError(
                    "artifact_path must be an existing run directory matching run_id",
                    code="RESULT_STORE_ARTIFACT_PATH_ERROR",
                )
            resolved_artifact = str(artifact)
        value = redact_secrets(
            StoredRun(
                project_id=project_id,
                run=run,
                artifact_path=resolved_artifact,
            ).model_dump(mode="json")
        )
        return StoredRun.model_validate_json(json.dumps(value))

    @staticmethod
    def _parse_stored(payload: str) -> StoredRun:
        try:
            return StoredRun.model_validate_json(payload)
        except ValidationError:
            raise ResultStoreError(
                "stored run does not match the expected schema",
                code="RESULT_STORE_FORMAT_ERROR",
            ) from None

    def _save_run_sync(
        self,
        run: RunResult,
        project_id: str,
        artifact_path: str | Path | None,
    ) -> StoredRun:
        self._initialize_sync()
        project_id = validate_project_id(project_id)
        validate_run_id(run.run_id)
        stored = self._prepare_stored(run, project_id, artifact_path)
        run_json = _json(stored.model_dump(mode="json"))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT run_json FROM runs WHERE run_id = ?", (run.run_id,)
            ).fetchone()
            if existing is not None:
                if existing["run_json"] != run_json:
                    raise ResultStoreError(
                        "run_id already exists with different evidence",
                        code="RESULT_STORE_CONFLICT",
                    )
                connection.rollback()
                return self._parse_stored(cast(str, existing["run_json"]))

            now = to_utc_iso8601(utc_now())
            connection.execute(
                "INSERT OR IGNORE INTO projects(project_id, name, created_at) VALUES(?, ?, ?)",
                (project_id, project_id, now),
            )
            mutation_json = (
                None
                if stored.run.mutation_score is None
                else _json(stored.run.mutation_score.model_dump(mode="json"))
            )
            connection.execute(
                """INSERT INTO runs(
                    run_id, project_id, status, exit_code, started_at, ended_at,
                    summary_json, mutation_score_json, run_json, artifact_path, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stored.run.run_id,
                    project_id,
                    stored.run.status,
                    stored.run.exit_code,
                    to_utc_iso8601(stored.run.started_at),
                    to_utc_iso8601(stored.run.ended_at),
                    _json(stored.run.summary.model_dump(mode="json")),
                    mutation_json,
                    run_json,
                    stored.artifact_path,
                    now,
                ),
            )
            for result in stored.run.results:
                connection.execute(
                    """INSERT INTO test_results(
                        trial_id, run_id, test_case_id, passed, started_at, ended_at, result_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                    (
                        result.trial_id,
                        stored.run.run_id,
                        result.test_case_id,
                        int(result.passed),
                        to_utc_iso8601(result.started_at),
                        to_utc_iso8601(result.ended_at),
                        _json(result.model_dump(mode="json")),
                    ),
                )
                connection.execute(
                    """INSERT INTO traces(
                        trace_id, run_id, test_case_id, trial_id, status,
                        artifact_root, trace_ref
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                    (
                        result.trace_id,
                        stored.run.run_id,
                        result.test_case_id,
                        result.trial_id,
                        result.trace_status,
                        stored.artifact_path,
                        result.trace_ref,
                    ),
                )
            for finding in findings_for(stored.run):
                connection.execute(
                    """INSERT INTO findings(
                        finding_id, run_id, trial_id, type, finding_json
                    ) VALUES(?, ?, ?, ?, ?)""",
                    (
                        finding.finding_id,
                        finding.run_id,
                        finding.trial_id,
                        finding.type,
                        _json(finding.model_dump(mode="json")),
                    ),
                )
            connection.commit()
            return stored
        except ResultStoreError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise ResultStoreError(
                f"SQLite transaction failed: {type(exc).__name__}",
                code="RESULT_STORE_DATABASE_ERROR",
            ) from None
        finally:
            connection.close()

    async def save_run(
        self,
        run: RunResult,
        *,
        project_id: str = "default",
        artifact_path: str | Path | None = None,
    ) -> StoredRun:
        return await asyncio.to_thread(
            self._save_run_sync,
            run,
            project_id,
            artifact_path,
        )

    def _get_run_sync(self, run_id: str) -> StoredRun | None:
        self._initialize_sync()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT run_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return None if row is None else self._parse_stored(cast(str, row["run_json"]))
        finally:
            connection.close()

    async def get_run(self, run_id: str) -> StoredRun | None:
        return await asyncio.to_thread(self._get_run_sync, run_id)

    def _list_runs_sync(self, project_id: str) -> list[StoredRunSummary]:
        self._initialize_sync()
        project_id = validate_project_id(project_id)
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT r.run_json, COUNT(f.finding_id) AS finding_count
                   FROM runs AS r
                   LEFT JOIN findings AS f ON f.run_id = r.run_id
                   WHERE r.project_id = ?
                   GROUP BY r.run_id
                   ORDER BY r.started_at DESC, r.run_id DESC""",
                (project_id,),
            ).fetchall()
            return [
                self._parse_stored(cast(str, row["run_json"])).to_summary(
                    finding_count=int(row["finding_count"])
                )
                for row in rows
            ]
        finally:
            connection.close()

    async def list_runs(self, project_id: str = "default") -> list[StoredRunSummary]:
        return await asyncio.to_thread(self._list_runs_sync, project_id)

    def _list_results_sync(self, run_id: str) -> list[TestResult]:
        self._initialize_sync()
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT result_json FROM test_results
                   WHERE run_id = ? ORDER BY test_case_id, trial_id""",
                (run_id,),
            ).fetchall()
            try:
                return [
                    TestResult.model_validate_json(cast(str, row["result_json"])) for row in rows
                ]
            except ValidationError:
                raise ResultStoreError(
                    "stored test result does not match the expected schema",
                    code="RESULT_STORE_FORMAT_ERROR",
                ) from None
        finally:
            connection.close()

    async def list_results(self, run_id: str) -> list[TestResult]:
        return await asyncio.to_thread(self._list_results_sync, run_id)

    def _get_trace_sync(self, trace_id: str) -> TraceRecord | None:
        self._initialize_sync()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT artifact_root, trace_ref FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if row is None or row["artifact_root"] is None or row["trace_ref"] is None:
                return None
            return read_trace_artifact(cast(str, row["artifact_root"]), cast(str, row["trace_ref"]))
        finally:
            connection.close()

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        return await asyncio.to_thread(self._get_trace_sync, trace_id)

    def _list_findings_sync(self, run_id: str) -> list[StoredFinding]:
        self._initialize_sync()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT finding_json FROM findings WHERE run_id = ? ORDER BY finding_id",
                (run_id,),
            ).fetchall()
            try:
                return [
                    StoredFinding.model_validate_json(cast(str, row["finding_json"]))
                    for row in rows
                ]
            except ValidationError:
                raise ResultStoreError(
                    "stored finding does not match the expected schema",
                    code="RESULT_STORE_FORMAT_ERROR",
                ) from None
        finally:
            connection.close()

    async def list_findings(self, run_id: str) -> list[StoredFinding]:
        return await asyncio.to_thread(self._list_findings_sync, run_id)

    def _get_review_sync(self, finding_id: str) -> FindingReview | None:
        self._initialize_sync()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT review_json FROM review_status WHERE finding_id = ?",
                (finding_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                return FindingReview.model_validate_json(cast(str, row["review_json"]))
            except ValidationError:
                raise ResultStoreError(
                    "stored review does not match the expected schema",
                    code="RESULT_STORE_FORMAT_ERROR",
                ) from None
        finally:
            connection.close()

    async def get_review(self, finding_id: str) -> FindingReview | None:
        return await asyncio.to_thread(self._get_review_sync, finding_id)

    def _save_review_sync(self, review: FindingReview) -> None:
        self._initialize_sync()
        connection = self._connect()
        try:
            value = redact_secrets(review.model_dump(mode="json"))
            connection.execute(
                """INSERT INTO review_status(
                    finding_id, status, note, updated_at, review_json
                ) VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(finding_id) DO UPDATE SET
                    status = excluded.status,
                    note = excluded.note,
                    updated_at = excluded.updated_at,
                    review_json = excluded.review_json""",
                (
                    review.finding_id,
                    review.status,
                    review.note,
                    to_utc_iso8601(review.updated_at),
                    _json(value),
                ),
            )
            connection.commit()
        except sqlite3.IntegrityError:
            connection.rollback()
            raise ResultStoreError(
                "finding does not exist", code="RESULT_STORE_NOT_FOUND"
            ) from None
        except sqlite3.Error as exc:
            connection.rollback()
            raise ResultStoreError(
                f"SQLite review update failed: {type(exc).__name__}",
                code="RESULT_STORE_DATABASE_ERROR",
            ) from None
        finally:
            connection.close()

    async def save_review(self, review: FindingReview) -> None:
        await asyncio.to_thread(self._save_review_sync, review)

    async def close(self) -> None:
        return None
