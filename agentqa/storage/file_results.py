"""Checksummed file implementation of the ResultStore protocol."""

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from agentqa.models import RunResult, TestResult, TraceRecord
from agentqa.models.storage import FindingReview, StoredFinding, StoredRun, StoredRunSummary
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.storage.redaction import redact_secrets
from agentqa.storage.results import (
    RESULT_STORE_SCHEMA_VERSION,
    ResultStoreError,
    finding_filename,
    findings_for,
    read_trace_artifact,
    validate_project_id,
    validate_run_id,
)


class FileResultStore:
    """Persist small run records while retaining large evidence by path."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._initialized = False
        self._index: dict[str, dict[str, Any]] = {
            "runs": {},
            "traces": {},
            "findings": {},
        }

    @staticmethod
    def _replace(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def _write_document(cls, path: Path, value: dict[str, Any]) -> None:
        document = {"schema_version": RESULT_STORE_SCHEMA_VERSION, **value}
        payload = canonical_json_bytes(document) + b"\n"
        cls._replace(path, payload)
        cls._replace(
            path.with_name(f"{path.name}.sha256"),
            f"{sha256_digest(payload)}\n".encode(),
        )

    @staticmethod
    def _read_document(path: Path) -> dict[str, Any]:
        checksum = path.with_name(f"{path.name}.sha256")
        try:
            payload = path.read_bytes()
            expected = checksum.read_text(encoding="utf-8").strip()
        except OSError:
            raise ResultStoreError(
                f"stored document is missing: {path.name}",
                code="RESULT_STORE_NOT_FOUND",
            ) from None
        if expected != sha256_digest(payload):
            raise ResultStoreError(
                f"stored document checksum mismatch: {path.name}",
                code="RESULT_STORE_CHECKSUM_ERROR",
            )
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            raise ResultStoreError(
                f"stored document is invalid JSON: {path.name}",
                code="RESULT_STORE_FORMAT_ERROR",
            ) from None
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != RESULT_STORE_SCHEMA_VERSION
        ):
            raise ResultStoreError(
                f"stored document has an incompatible schema: {path.name}",
                code="RESULT_STORE_FORMAT_ERROR",
            )
        value.pop("schema_version")
        return cast(dict[str, Any], value)

    def _record_path(self, project_id: str, run_id: str) -> Path:
        return self.root / "projects" / project_id / "runs" / f"{run_id}.json"

    def _review_path(self, finding_id: str) -> Path:
        return self.root / "reviews" / finding_filename(finding_id)

    @staticmethod
    def _parse_record(document: dict[str, Any]) -> StoredRun:
        try:
            return StoredRun.model_validate_json(json.dumps(document["stored_run"]))
        except (KeyError, ValidationError):
            raise ResultStoreError(
                "stored run does not match the expected schema",
                code="RESULT_STORE_FORMAT_ERROR",
            ) from None

    def _rebuild_index(self) -> None:
        index: dict[str, dict[str, Any]] = {"runs": {}, "traces": {}, "findings": {}}
        records_root = self.root / "projects"
        if records_root.exists():
            for path in sorted(records_root.glob("*/runs/*.json")):
                stored = self._parse_record(self._read_document(path))
                relative = path.relative_to(self.root).as_posix()
                existing = index["runs"].get(stored.run.run_id)
                if existing is not None and existing["path"] != relative:
                    raise ResultStoreError(
                        "duplicate run_id exists in multiple projects",
                        code="RESULT_STORE_CONFLICT",
                    )
                index["runs"][stored.run.run_id] = {
                    "project_id": stored.project_id,
                    "path": relative,
                }
                for finding in findings_for(stored.run):
                    index["findings"][finding.finding_id] = stored.run.run_id
                if stored.artifact_path is not None:
                    for result in stored.run.results:
                        if result.trace_ref is not None:
                            index["traces"][result.trace_id] = {
                                "run_id": stored.run.run_id,
                                "artifact_path": stored.artifact_path,
                                "trace_ref": result.trace_ref,
                            }
        self._index = index
        self._write_document(self.root / "index.json", index)

    def _initialize_sync(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            self._rebuild_index()
            self._initialized = True

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    @staticmethod
    def _stored_run(
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
        redacted = redact_secrets(
            StoredRun(
                project_id=project_id,
                run=run,
                artifact_path=resolved_artifact,
            ).model_dump(mode="json")
        )
        return StoredRun.model_validate_json(json.dumps(redacted))

    def _save_run_sync(
        self,
        run: RunResult,
        project_id: str,
        artifact_path: str | Path | None,
    ) -> StoredRun:
        self._initialize_sync()
        project_id = validate_project_id(project_id)
        validate_run_id(run.run_id)
        stored = self._stored_run(run, project_id, artifact_path)
        path = self._record_path(project_id, run.run_id)
        with self._lock:
            existing_index = self._index["runs"].get(run.run_id)
            if existing_index is not None:
                existing_path = self.root / str(existing_index["path"])
                existing = self._parse_record(self._read_document(existing_path))
                if existing != stored:
                    raise ResultStoreError(
                        "run_id already exists with different evidence",
                        code="RESULT_STORE_CONFLICT",
                    )
                return existing
            self._write_document(path, {"stored_run": stored.model_dump(mode="json")})
            self._rebuild_index()
        return stored

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
        with self._lock:
            entry = self._index["runs"].get(run_id)
            if entry is None:
                return None
            return self._parse_record(self._read_document(self.root / str(entry["path"])))

    async def get_run(self, run_id: str) -> StoredRun | None:
        return await asyncio.to_thread(self._get_run_sync, run_id)

    def _list_runs_sync(self, project_id: str) -> list[StoredRunSummary]:
        project_id = validate_project_id(project_id)
        self._initialize_sync()
        stored_runs = [
            self._parse_record(self._read_document(self.root / str(entry["path"])))
            for entry in self._index["runs"].values()
            if entry["project_id"] == project_id
        ]
        summaries = [
            stored.to_summary(finding_count=len(findings_for(stored.run))) for stored in stored_runs
        ]
        return sorted(summaries, key=lambda item: (item.started_at, item.run_id), reverse=True)

    async def list_runs(self, project_id: str = "default") -> list[StoredRunSummary]:
        return await asyncio.to_thread(self._list_runs_sync, project_id)

    async def list_results(self, run_id: str) -> list[TestResult]:
        stored = await self.get_run(run_id)
        return [] if stored is None else list(stored.run.results)

    def _get_trace_sync(self, trace_id: str) -> TraceRecord | None:
        self._initialize_sync()
        entry = self._index["traces"].get(trace_id)
        if entry is None:
            return None
        return read_trace_artifact(str(entry["artifact_path"]), str(entry["trace_ref"]))

    async def get_trace(self, trace_id: str) -> TraceRecord | None:
        return await asyncio.to_thread(self._get_trace_sync, trace_id)

    async def list_findings(self, run_id: str) -> list[StoredFinding]:
        stored = await self.get_run(run_id)
        return [] if stored is None else findings_for(stored.run)

    def _get_review_sync(self, finding_id: str) -> FindingReview | None:
        self._initialize_sync()
        path = self._review_path(finding_id)
        if not path.exists():
            return None
        try:
            return FindingReview.model_validate_json(
                json.dumps(self._read_document(path)["review"])
            )
        except (KeyError, ValidationError):
            raise ResultStoreError(
                "stored review does not match the expected schema",
                code="RESULT_STORE_FORMAT_ERROR",
            ) from None

    async def get_review(self, finding_id: str) -> FindingReview | None:
        return await asyncio.to_thread(self._get_review_sync, finding_id)

    def _save_review_sync(self, review: FindingReview) -> None:
        self._initialize_sync()
        with self._lock:
            if review.finding_id not in self._index["findings"]:
                raise ResultStoreError("finding does not exist", code="RESULT_STORE_NOT_FOUND")
            self._write_document(
                self._review_path(review.finding_id),
                {"review": cast(dict[str, Any], redact_secrets(review.model_dump(mode="json")))},
            )

    async def save_review(self, review: FindingReview) -> None:
        await asyncio.to_thread(self._save_review_sync, review)

    async def close(self) -> None:
        return None
