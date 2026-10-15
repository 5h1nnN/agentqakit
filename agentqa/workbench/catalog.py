"""Verified workspace artifact catalog and explicit import service."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import sqlite3
import threading
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import ValidationError

from agentqa.comparison import ComparisonError, load_run_artifact
from agentqa.conventions import utc_now
from agentqa.errors import AgentQAError
from agentqa.models import (
    AnalysisReport,
    ComparisonReport,
    GateResult,
    GenerationPlan,
    GenerationReport,
    RunResult,
    TraceRecord,
)
from agentqa.models.generation import CodeEvidence
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.storage import ResultStore
from agentqa.workbench.models import (
    ArtifactImportIssue,
    ArtifactImportResult,
    CatalogArtifact,
    HandoffBundle,
    WorkflowReport,
)

CATALOG_SCHEMA_VERSION = "agentqa.artifact-catalog/v1alpha1"
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_IMPORT_FILES = 5_000
_MAX_ISSUES = 100
ArtifactSource = Literal["CLI", "API", "MCP", "WORKFLOW", "BENCHMARK", "REFERENCE", "IMPORTED"]
DiscoverySource = Literal["artifacts", "mcp", "benchmark_results", "reference_results"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_catalog (
    project_id TEXT NOT NULL,
    path TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    checksum TEXT NOT NULL,
    artifact_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(project_id, path)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_catalog_artifact_identity
    ON artifact_catalog(project_id, artifact_id);
CREATE INDEX IF NOT EXISTS idx_catalog_project_type
    ON artifact_catalog(project_id, artifact_type, updated_at DESC);
CREATE TABLE IF NOT EXISTS selected_code_evidence (
    project_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    path TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(project_id, content_hash)
);
"""


class ArtifactCatalogError(AgentQAError):
    def __init__(self, message: str, *, code: str = "ARTIFACT_CATALOG_ERROR") -> None:
        super().__init__(message, code=code)


def _json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _safe_relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError:
        raise ArtifactCatalogError(
            "artifact path escapes the fixed workspace", code="ARTIFACT_CATALOG_PATH_ERROR"
        ) from None


def _contains_symlink(path: Path, workspace: Path) -> bool:
    """Check every lexical component before resolve() erases symlink identity."""
    try:
        relative = path.absolute().relative_to(workspace.absolute())
    except ValueError:
        return True
    cursor = workspace
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            return True
    return False


def _without_artifact_version(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if result.get("schema_version") == "agentqa.artifact/v1alpha1":
        result.pop("schema_version")
    return result


def _artifact_type(path: Path) -> str:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if name == "generation-report.json":
        return "GenerationReport"
    if name in {"plan.json", "generation-plan.json"}:
        return "GenerationPlan"
    if "analysis" in name or parts.intersection({"analysis", "analyses"}):
        return "AnalysisReport"
    if name == "comparison.json":
        return "ComparisonReport"
    if name == "handoff-bundle.json":
        return "HandoffBundle"
    if name == "workflow-report.json":
        return "WorkflowReport"
    if "gate" in name and name.endswith(".json"):
        return "GateResult"
    if name == "mcp-transcript.json":
        return "MCPTranscript"
    if name == "run.json":
        return "RunLifecycle"
    if name == "summary.json":
        return "RunSummary" if "runs" in parts else "BenchmarkSummary"
    if name == "results.jsonl":
        return "TestResults"
    if "traces" in parts and name.endswith(".json"):
        return "Trace"
    if "cassettes" in parts or "cassette" in name:
        return "ReplayCassette"
    if name.endswith(".zip"):
        return "FailureBundle"
    if name.endswith(".csv"):
        return "Metrics"
    if name == "environment.json":
        return "Environment"
    if name == "scores.json":
        return "AnalysisScores"
    return "JSONDocument" if name.endswith(".json") else "Artifact"


def _source_for(path: str, requested: DiscoverySource | None) -> ArtifactSource:
    if requested == "mcp" or path.startswith(".agentqa/mcp/"):
        return "MCP"
    if requested == "reference_results" or path.startswith("benchmark/reference-results/"):
        return "REFERENCE"
    if requested == "benchmark_results" or path.startswith("benchmark/results/"):
        return "BENCHMARK"
    if requested == "artifacts":
        return "CLI"
    return "IMPORTED"


def _validate_payload(path: Path, payload: bytes, artifact_type: str) -> tuple[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        if not zipfile.is_zipfile(path):
            raise ArtifactCatalogError(
                "failure bundle is not a valid ZIP file", code="ARTIFACT_CATALOG_SCHEMA_ERROR"
            )
        return "application/zip", {}
    if suffix == ".csv":
        try:
            rows = csv.reader(io.StringIO(payload.decode("utf-8")))
            header = next(rows)
        except (UnicodeError, csv.Error, StopIteration):
            raise ArtifactCatalogError(
                "CSV artifact has no valid header", code="ARTIFACT_CATALOG_SCHEMA_ERROR"
            ) from None
        if not header or any(not value.strip() for value in header):
            raise ArtifactCatalogError(
                "CSV artifact has an invalid header", code="ARTIFACT_CATALOG_SCHEMA_ERROR"
            )
        return "text/csv", {"columns": len(header)}
    if suffix == ".jsonl":
        count = 0
        try:
            for line in payload.decode("utf-8").splitlines():
                if line.strip():
                    json.loads(line)
                    count += 1
        except (UnicodeError, json.JSONDecodeError):
            raise ArtifactCatalogError(
                "JSONL artifact is malformed", code="ARTIFACT_CATALOG_SCHEMA_ERROR"
            ) from None
        return "application/jsonl", {"records": count}
    if suffix != ".json":
        raise ArtifactCatalogError(
            "artifact type is not supported by the catalog",
            code="ARTIFACT_CATALOG_TYPE_ERROR",
        )
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError):
        raise ArtifactCatalogError(
            "JSON artifact is malformed", code="ARTIFACT_CATALOG_SCHEMA_ERROR"
        ) from None
    if not isinstance(value, dict):
        raise ArtifactCatalogError(
            "JSON artifact must contain an object", code="ARTIFACT_CATALOG_SCHEMA_ERROR"
        )
    document = cast(dict[str, Any], value)
    try:
        if artifact_type == "GenerationReport":
            GenerationReport.model_validate_json(payload)
        elif artifact_type == "GenerationPlan":
            GenerationPlan.model_validate_json(payload)
        elif artifact_type == "AnalysisReport":
            AnalysisReport.model_validate_json(payload)
        elif artifact_type == "ComparisonReport":
            ComparisonReport.model_validate_json(payload)
        elif artifact_type == "HandoffBundle":
            HandoffBundle.model_validate_json(payload)
        elif artifact_type == "WorkflowReport":
            WorkflowReport.model_validate_json(payload)
        elif artifact_type == "GateResult":
            GateResult.model_validate_json(payload)
        elif artifact_type == "Trace":
            TraceRecord.model_validate_json(
                json.dumps(_without_artifact_version(document), ensure_ascii=False)
            )
        elif artifact_type == "RunLifecycle":
            if not isinstance(document.get("run_id"), str) or not isinstance(
                document.get("status"), str
            ):
                raise ValueError("run lifecycle fields are missing")
        elif artifact_type == "MCPTranscript" and not isinstance(
            document.get("schema_version"), str
        ):
            raise ValueError("MCP transcript schema version is missing")
    except (ValidationError, ValueError):
        raise ArtifactCatalogError(
            f"{artifact_type} does not match its expected schema",
            code="ARTIFACT_CATALOG_SCHEMA_ERROR",
        ) from None
    schema_version = document.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        schema_version = f"agentqa.{artifact_type.lower()}/v1alpha1"
    metadata: dict[str, Any] = {}
    for key in ("run_id", "generation_id", "analysis_id", "passed", "exit_code"):
        if isinstance(document.get(key), str | int | float | bool):
            metadata[key] = document[key]
    return schema_version, metadata


class ArtifactCatalogService:
    """Index verified files without moving or rewriting authoritative artifacts."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        workspace_root: str | Path,
        result_store: ResultStore,
        artifact_directory: str | Path,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.result_store = result_store
        self.artifact_directory = Path(artifact_directory).resolve()
        self._init_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize_sync(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(_SCHEMA)
                row = connection.execute(
                    "SELECT value FROM schema_metadata "
                    "WHERE key = 'artifact_catalog_schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO schema_metadata(key, value) VALUES(?, ?)",
                        ("artifact_catalog_schema_version", CATALOG_SCHEMA_VERSION),
                    )
                elif row["value"] != CATALOG_SCHEMA_VERSION:
                    raise ArtifactCatalogError(
                        "artifact catalog schema is not supported",
                        code="ARTIFACT_CATALOG_VERSION_ERROR",
                    )
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    async def close(self) -> None:
        return None

    def _roots(self, sources: Iterable[DiscoverySource]) -> list[tuple[DiscoverySource, Path]]:
        mapping: dict[DiscoverySource, list[Path]] = {
            "artifacts": [
                self.artifact_directory,
                self.workspace_root / ".agentqa" / "artifacts",
                self.workspace_root / "artifacts",
            ],
            "mcp": [self.workspace_root / ".agentqa" / "mcp"],
            "benchmark_results": [self.workspace_root / "benchmark" / "results"],
            "reference_results": [self.workspace_root / "benchmark" / "reference-results"],
        }
        roots: list[tuple[DiscoverySource, Path]] = []
        seen: set[Path] = set()
        for source in sources:
            for root in mapping[source]:
                try:
                    resolved = root.resolve()
                    resolved.relative_to(self.workspace_root)
                except ValueError:
                    continue
                if resolved not in seen and resolved.is_dir():
                    seen.add(resolved)
                    roots.append((source, resolved))
        return roots

    def _candidate(self, project_id: str, path: Path, source: ArtifactSource) -> CatalogArtifact:
        relative = _safe_relative(path, self.workspace_root)
        if _contains_symlink(path, self.workspace_root):
            raise ArtifactCatalogError(
                "symlink artifacts are not imported", code="ARTIFACT_CATALOG_PATH_ERROR"
            )
        size = path.stat().st_size
        if size > _MAX_ARTIFACT_BYTES:
            raise ArtifactCatalogError(
                "artifact exceeds the 16 MiB import limit",
                code="ARTIFACT_CATALOG_SIZE_ERROR",
            )
        checksum_path = path.with_name(f"{path.name}.sha256")
        if not checksum_path.is_file() or checksum_path.is_symlink():
            raise ArtifactCatalogError(
                "artifact checksum sidecar is missing",
                code="ARTIFACT_CATALOG_CHECKSUM_ERROR",
            )
        payload = path.read_bytes()
        checksum = sha256_digest(payload)
        try:
            expected = checksum_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            expected = ""
        if expected != checksum:
            raise ArtifactCatalogError(
                "artifact checksum does not match its sidecar",
                code="ARTIFACT_CATALOG_CHECKSUM_ERROR",
            )
        artifact_type = _artifact_type(path)
        schema_version, metadata = _validate_payload(path, payload, artifact_type)
        parts = PurePosixPath(relative).parts
        run_id = None
        if "runs" in parts:
            index = parts.index("runs")
            if index + 1 < len(parts):
                run_id = parts[index + 1]
        identity = sha256_digest(
            canonical_json_bytes({"project_id": project_id, "path": relative, "checksum": checksum})
        )
        return CatalogArtifact(
            artifact_id=identity,
            project_id=project_id,
            artifact_type=artifact_type,
            source=source,
            path=relative,
            schema_version=schema_version,
            checksum=checksum,
            size_bytes=size,
            created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
            run_id=run_id,
            metadata=metadata,
        )

    def _upsert_sync(self, artifact: CatalogArtifact) -> bool:
        self._initialize_sync()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT artifact_json FROM artifact_catalog WHERE project_id = ? AND path = ?",
                (artifact.project_id, artifact.path),
            ).fetchone()
            unchanged = False
            if row is not None:
                try:
                    previous = CatalogArtifact.model_validate_json(row["artifact_json"])
                    unchanged = previous == artifact
                except ValidationError:
                    unchanged = False
            connection.execute(
                """INSERT INTO artifact_catalog(
                       project_id, path, artifact_id, artifact_type, checksum,
                       artifact_json, updated_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(project_id, path) DO UPDATE SET
                       artifact_id = excluded.artifact_id,
                       artifact_type = excluded.artifact_type,
                       checksum = excluded.checksum,
                       artifact_json = excluded.artifact_json,
                       updated_at = excluded.updated_at""",
                (
                    artifact.project_id,
                    artifact.path,
                    artifact.artifact_id,
                    artifact.artifact_type,
                    artifact.checksum,
                    _json(artifact.model_dump(mode="json")),
                    utc_now().isoformat(),
                ),
            )
            connection.commit()
            return unchanged
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ArtifactCatalogError(
                f"artifact catalog transaction failed: {type(exc).__name__}",
                code="ARTIFACT_CATALOG_DATABASE_ERROR",
            ) from None
        finally:
            connection.close()

    async def register_path(
        self,
        project_id: str,
        path: Path,
        *,
        source: ArtifactSource,
        workflow_id: str | None = None,
    ) -> CatalogArtifact:
        artifact = await asyncio.to_thread(self._candidate, project_id, path, source)
        if workflow_id is not None:
            artifact = artifact.model_copy(update={"workflow_id": workflow_id})
        await asyncio.to_thread(self._upsert_sync, artifact)
        return artifact

    async def register_directory(
        self, project_id: str, directory: Path, *, source: ArtifactSource
    ) -> list[CatalogArtifact]:
        resolved = directory.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise ArtifactCatalogError(
                "artifact directory escapes the fixed workspace",
                code="ARTIFACT_CATALOG_PATH_ERROR",
            ) from None
        registered: list[CatalogArtifact] = []
        for path in sorted(resolved.rglob("*")):
            if (
                not path.is_file()
                or path.name.endswith(".sha256")
                or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".zip"}
            ):
                continue
            artifact = await self.register_path(project_id, path, source=source)
            registered.append(artifact)
        return registered

    def _scan_sync(
        self, project_id: str, sources: list[DiscoverySource]
    ) -> tuple[list[CatalogArtifact], list[ArtifactImportIssue], int, list[Path]]:
        artifacts: list[CatalogArtifact] = []
        issues: list[ArtifactImportIssue] = []
        scanned = 0
        run_directories: set[Path] = set()
        for source_name, root in self._roots(sources):
            for path in sorted(root.rglob("*")):
                if scanned >= _MAX_IMPORT_FILES:
                    if len(issues) < _MAX_ISSUES:
                        issues.append(
                            ArtifactImportIssue(
                                path=root.relative_to(self.workspace_root).as_posix(),
                                code="ARTIFACT_IMPORT_LIMIT",
                                message="artifact discovery stopped at 5000 files",
                            )
                        )
                    return artifacts, issues, scanned, sorted(run_directories)
                if not path.is_file() or path.name.endswith(".sha256"):
                    continue
                if path.parent.parent.name == "runs":
                    run_directories.add(path.parent)
                if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".zip"}:
                    continue
                scanned += 1
                relative = _safe_relative(path, self.workspace_root)
                try:
                    artifacts.append(
                        self._candidate(project_id, path, _source_for(relative, source_name))
                    )
                except (ArtifactCatalogError, OSError) as exc:
                    if len(issues) < _MAX_ISSUES:
                        issues.append(
                            ArtifactImportIssue(
                                path=relative,
                                code=(
                                    exc.code
                                    if isinstance(exc, ArtifactCatalogError)
                                    else "ARTIFACT_CATALOG_READ_ERROR"
                                ),
                                message=(
                                    exc.message
                                    if isinstance(exc, ArtifactCatalogError)
                                    else f"artifact could not be read: {type(exc).__name__}"
                                ),
                            )
                        )
        return artifacts, issues, scanned, sorted(run_directories)

    async def discover(
        self, project_id: str, sources: list[DiscoverySource]
    ) -> ArtifactImportResult:
        artifacts, issues, scanned, run_directories = await asyncio.to_thread(
            self._scan_sync, project_id, sources
        )
        imported = 0
        unchanged = 0
        for artifact in artifacts:
            if await asyncio.to_thread(self._upsert_sync, artifact):
                unchanged += 1
            else:
                imported += 1
        runs_registered = 0
        for directory in run_directories:
            try:
                run_artifact = await asyncio.to_thread(load_run_artifact, directory)
                if await self.result_store.get_run(run_artifact.run_id) is not None:
                    continue
                if run_artifact.results:
                    started_at = min(result.started_at for result in run_artifact.results)
                    ended_at = max(result.ended_at for result in run_artifact.results)
                else:
                    stamp = datetime.fromtimestamp(directory.stat().st_mtime, tz=UTC)
                    started_at = stamp
                    ended_at = stamp
                run = RunResult(
                    run_id=run_artifact.run_id,
                    status="COMPLETED",
                    exit_code=0 if run_artifact.summary.failed == 0 else 1,
                    started_at=started_at,
                    ended_at=ended_at,
                    summary=run_artifact.summary,
                    results=list(run_artifact.results),
                    mutation_score=run_artifact.mutation_score,
                )
                await self.result_store.save_run(
                    run, project_id=project_id, artifact_path=directory
                )
                runs_registered += 1
            except (ComparisonError, OSError, ValidationError) as exc:
                if len(issues) < _MAX_ISSUES:
                    issues.append(
                        ArtifactImportIssue(
                            path=_safe_relative(directory, self.workspace_root),
                            code=(
                                exc.code
                                if isinstance(exc, ComparisonError)
                                else "ARTIFACT_RUN_IMPORT_ERROR"
                            ),
                            message=(
                                exc.message
                                if isinstance(exc, ComparisonError)
                                else f"run could not be imported: {type(exc).__name__}"
                            ),
                        )
                    )
        return ArtifactImportResult(
            project_id=project_id,
            scanned=scanned,
            imported=imported,
            unchanged=unchanged,
            rejected=len(issues),
            runs_registered=runs_registered,
            issues=issues,
        )

    def _list_sync(
        self,
        project_id: str,
        artifact_type: str | None,
        limit: int | None,
        offset: int,
    ) -> list[CatalogArtifact]:
        self._initialize_sync()
        connection = self._connect()
        try:
            query = "SELECT artifact_json FROM artifact_catalog WHERE project_id = ?"
            parameters: list[str | int] = [project_id]
            if artifact_type is not None:
                query += " AND artifact_type = ?"
                parameters.append(artifact_type)
            query += " ORDER BY updated_at DESC, path"
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                parameters.extend((limit, offset))
            rows = connection.execute(query, parameters).fetchall()
            artifacts: list[CatalogArtifact] = []
            for row in rows:
                artifact = CatalogArtifact.model_validate_json(row["artifact_json"])
                path = self.workspace_root / PurePosixPath(artifact.path)
                if not path.is_file():
                    availability = "MISSING"
                elif _contains_symlink(path, self.workspace_root):
                    availability = "INVALID"
                else:
                    try:
                        expected = (
                            path.with_name(f"{path.name}.sha256")
                            .read_text(encoding="utf-8")
                            .strip()
                        )
                        actual = sha256_digest(path.read_bytes())
                        availability = (
                            "AVAILABLE" if expected == artifact.checksum == actual else "INVALID"
                        )
                    except (OSError, UnicodeError):
                        availability = "INVALID"
                artifacts.append(artifact.model_copy(update={"availability": availability}))
            return artifacts
        finally:
            connection.close()

    async def list_artifacts(
        self,
        project_id: str,
        artifact_type: str | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[CatalogArtifact]:
        return await asyncio.to_thread(
            self._list_sync,
            project_id,
            artifact_type,
            limit,
            offset,
        )

    def _count_sync(self, project_id: str, artifact_type: str | None) -> int:
        self._initialize_sync()
        connection = self._connect()
        try:
            if artifact_type is None:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM artifact_catalog WHERE project_id = ?",
                    (project_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM artifact_catalog "
                    "WHERE project_id = ? AND artifact_type = ?",
                    (project_id, artifact_type),
                ).fetchone()
            return 0 if row is None else int(row["total"])
        finally:
            connection.close()

    async def count_artifacts(self, project_id: str, artifact_type: str | None = None) -> int:
        return await asyncio.to_thread(self._count_sync, project_id, artifact_type)

    def _add_code_evidence_sync(self, project_id: str, evidence: CodeEvidence) -> CodeEvidence:
        self._initialize_sync()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT evidence_json FROM selected_code_evidence WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            existing = [CodeEvidence.model_validate_json(row["evidence_json"]) for row in rows]
            match = next(
                (item for item in existing if item.content_hash == evidence.content_hash), None
            )
            if match is not None:
                if match != evidence:
                    raise ArtifactCatalogError(
                        "code evidence hash already identifies a different selection",
                        code="CODE_EVIDENCE_CONFLICT",
                    )
                return match
            if len(existing) >= 10:
                raise ArtifactCatalogError(
                    "a project can select at most 10 code evidence snippets",
                    code="CODE_EVIDENCE_LIMIT",
                )
            if (
                sum(len(item.content.encode("utf-8")) for item in existing)
                + len(evidence.content.encode("utf-8"))
                > 160 * 1024
            ):
                raise ArtifactCatalogError(
                    "selected code evidence exceeds 163840 bytes",
                    code="CODE_EVIDENCE_LIMIT",
                )
            connection.execute(
                """INSERT INTO selected_code_evidence(
                       project_id, content_hash, path, evidence_json, created_at
                   ) VALUES(?, ?, ?, ?, ?)""",
                (
                    project_id,
                    evidence.content_hash,
                    evidence.path,
                    _json(evidence.model_dump(mode="json", by_alias=True)),
                    utc_now().isoformat(),
                ),
            )
            connection.commit()
            return evidence
        finally:
            connection.close()

    async def add_code_evidence(self, project_id: str, evidence: CodeEvidence) -> CodeEvidence:
        return await asyncio.to_thread(self._add_code_evidence_sync, project_id, evidence)

    def _list_code_evidence_sync(self, project_id: str) -> list[CodeEvidence]:
        self._initialize_sync()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT evidence_json FROM selected_code_evidence WHERE project_id = ? "
                "ORDER BY created_at, path",
                (project_id,),
            ).fetchall()
            return [CodeEvidence.model_validate_json(row["evidence_json"]) for row in rows]
        finally:
            connection.close()

    async def list_code_evidence(self, project_id: str) -> list[CodeEvidence]:
        return await asyncio.to_thread(self._list_code_evidence_sync, project_id)


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "ArtifactCatalogError",
    "ArtifactCatalogService",
    "ArtifactSource",
    "DiscoverySource",
]
