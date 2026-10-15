"""FastAPI composition root and HTTP routes."""

import gzip
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agentqa.api.ai_evidence import list_ai_evidence
from agentqa.api.jobs import RunJobManager
from agentqa.api.models import (
    ArtifactImportRequest,
    CodeEvidenceRequest,
    ComparisonRequest,
    GateEvaluateRequest,
    ProjectCreateRequest,
    ReviewUpdateRequest,
    RunCreateRequest,
    RunJob,
    WorkflowCreateRequest,
)
from agentqa.comparison import ComparisonError, compare_runs, load_run_artifact
from agentqa.conventions import utc_now
from agentqa.evaluators import (
    JudgeProviderLike,
    configured_deepseek_provider,
    load_judge_environment,
)
from agentqa.gates import GateError, evaluate_gate
from agentqa.models import QualityGateSpec
from agentqa.models.storage import FindingReview
from agentqa.specs import SpecLoadError, load_spec_file
from agentqa.storage import ResultStore, ResultStoreError, SQLiteResultStore
from agentqa.tracing import InMemoryTraceStore, TraceOptions
from agentqa.tracing.otlp import OTLPDecodeError, ingest_otlp_protobuf
from agentqa.workbench import (
    ArtifactCatalogError,
    ArtifactCatalogService,
    CodeEvidenceError,
    CodeEvidenceService,
    EvidenceBoundaryService,
    ProjectPreflightService,
    SQLiteWorkflowStore,
    WorkflowOrchestrator,
    WorkflowStore,
    WorkflowStoreError,
)


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Local API paths and safety limits."""

    workspace_root: Path
    artifact_directory: Path
    database_path: Path
    static_directory: Path | None = None
    max_concurrent_runs: int = 2
    max_concurrent_ai_steps: int = 1
    max_concurrent_campaigns: int = 1
    max_receiver_bytes: int = 8 * 1024 * 1024

    @classmethod
    def defaults(cls) -> "ApiSettings":
        workspace = Path.cwd().resolve()
        return cls(
            workspace_root=workspace,
            artifact_directory=workspace / ".agentqa" / "artifacts",
            database_path=workspace / ".agentqa" / "agentqa.sqlite3",
            static_directory=workspace / "apps" / "web" / "dist",
        )


def _http_error(code: str, message: str, status_code: int) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def create_app(
    settings: ApiSettings | None = None,
    *,
    store: ResultStore | None = None,
    workflow_store: WorkflowStore | None = None,
    judge_provider: JudgeProviderLike | None = None,
) -> FastAPI:
    """Build an injectable local API without global database connections."""
    settings = settings or ApiSettings.defaults()
    load_judge_environment(settings.workspace_root / ".env.local")
    active_judge = judge_provider or configured_deepseek_provider()
    result_store = store or SQLiteResultStore(settings.database_path)
    active_workflow_store = workflow_store or SQLiteWorkflowStore(settings.database_path)
    artifact_catalog = ArtifactCatalogService(
        settings.database_path,
        workspace_root=settings.workspace_root,
        result_store=result_store,
        artifact_directory=settings.artifact_directory,
    )
    code_evidence = CodeEvidenceService(settings.workspace_root)
    preflight = ProjectPreflightService(
        settings.workspace_root,
        active_workflow_store,
        active_judge,
    )
    evidence_boundary = EvidenceBoundaryService(
        settings.workspace_root,
        artifact_catalog,
        active_judge,
    )
    trace_options = TraceOptions(max_receiver_bytes=settings.max_receiver_bytes)
    live_traces = InMemoryTraceStore(max_spans_per_trace=trace_options.max_spans_per_trace)
    jobs = RunJobManager(
        result_store,
        workspace_root=settings.workspace_root,
        artifact_directory=settings.artifact_directory,
        artifact_catalog=artifact_catalog,
        judge_provider=active_judge,
        max_concurrent_runs=settings.max_concurrent_runs,
    )
    workflows = WorkflowOrchestrator(
        active_workflow_store,
        max_concurrent_ai_steps=settings.max_concurrent_ai_steps,
        max_concurrent_campaigns=settings.max_concurrent_campaigns,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        await result_store.initialize()
        await active_workflow_store.initialize()
        await artifact_catalog.initialize()
        try:
            yield
        finally:
            await workflows.shutdown()
            await jobs.shutdown()
            await active_workflow_store.close()
            await artifact_catalog.close()
            await result_store.close()

    app = FastAPI(
        title="AgentQAKit API",
        version="v1alpha1",
        lifespan=lifespan,
    )
    app.state.result_store = result_store
    app.state.workflow_store = active_workflow_store
    app.state.workflow_orchestrator = workflows
    app.state.artifact_catalog = artifact_catalog
    app.state.run_jobs = jobs
    app.state.live_traces = live_traces
    app.state.judge_provider = active_judge
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["content-type"],
    )

    @app.exception_handler(ResultStoreError)
    async def result_store_error_handler(request: Request, exc: ResultStoreError) -> Response:
        del request
        status_code = 404 if exc.code.endswith("NOT_FOUND") else 409
        return Response(
            content=json.dumps({"detail": exc.to_dict()}),
            status_code=status_code,
            media_type="application/json",
        )

    @app.exception_handler(WorkflowStoreError)
    async def workflow_store_error_handler(request: Request, exc: WorkflowStoreError) -> Response:
        del request
        status_code = 404 if exc.code.endswith("NOT_FOUND") else 409
        return Response(
            content=json.dumps({"detail": exc.to_dict()}),
            status_code=status_code,
            media_type="application/json",
        )

    @app.exception_handler(ArtifactCatalogError)
    async def artifact_catalog_error_handler(
        request: Request, exc: ArtifactCatalogError
    ) -> Response:
        del request
        status_code = 404 if exc.code.endswith(("NOT_FOUND", "MISSING")) else 409
        return Response(
            content=json.dumps({"detail": exc.to_dict()}),
            status_code=status_code,
            media_type="application/json",
        )

    @app.exception_handler(CodeEvidenceError)
    async def code_evidence_error_handler(request: Request, exc: CodeEvidenceError) -> Response:
        del request
        status_code = 404 if exc.code.endswith("MISSING") else 422
        return Response(
            content=json.dumps({"detail": exc.to_dict()}),
            status_code=status_code,
            media_type="application/json",
        )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "storage": type(result_store).__name__,
            "workflowStorage": type(active_workflow_store).__name__,
            "workspaceRoot": str(settings.workspace_root.resolve()),
            "judge": {
                "configured": active_judge is not None,
                "provider": None if active_judge is None else active_judge.name,
                "model": None if active_judge is None else active_judge.model,
            },
        }

    @app.get("/api/projects")
    async def list_projects() -> dict[str, Any]:
        return {"projects": await active_workflow_store.list_projects()}

    @app.post("/api/projects", status_code=status.HTTP_201_CREATED)
    async def create_project(request: ProjectCreateRequest) -> Any:
        return await active_workflow_store.create_project(
            request.project_id,
            request.name,
            request.description,
            workspace_path=request.workspace_path,
            default_spec_refs=request.default_spec_refs,
            tags=request.tags,
        )

    async def require_project(project_id: str) -> Any:
        project = await active_workflow_store.get_project(project_id)
        if project is None:
            raise _http_error("WORKBENCH_PROJECT_NOT_FOUND", "project does not exist", 404)
        return project

    @app.get("/api/projects/{project_id}")
    async def get_project(project_id: str) -> Any:
        return await require_project(project_id)

    @app.post("/api/projects/{project_id}/preflight")
    async def run_project_preflight(project_id: str) -> Any:
        return await preflight.run(await require_project(project_id))

    @app.post("/api/projects/{project_id}/artifacts/import")
    async def import_project_artifacts(project_id: str, request: ArtifactImportRequest) -> Any:
        await require_project(project_id)
        return await artifact_catalog.discover(project_id, request.sources)

    @app.get("/api/projects/{project_id}/artifacts")
    async def list_project_artifacts(
        project_id: str,
        artifact_type: str | None = None,
        limit: int = Query(default=200, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        await require_project(project_id)
        artifacts = await artifact_catalog.list_artifacts(
            project_id,
            artifact_type,
            limit=limit,
            offset=offset,
        )
        return {
            "artifacts": artifacts,
            "total": await artifact_catalog.count_artifacts(project_id, artifact_type),
            "limit": limit,
            "offset": offset,
        }

    @app.get("/api/projects/{project_id}/workspace")
    async def browse_project_workspace(
        project_id: str,
        path: str = Query(default=".", max_length=2_000),
    ) -> dict[str, Any]:
        project = await require_project(project_id)
        return {"entries": code_evidence.browse(project, path)}

    @app.post("/api/projects/{project_id}/code-evidence/preview")
    async def preview_code_evidence(project_id: str, request: CodeEvidenceRequest) -> Any:
        project = await require_project(project_id)
        return code_evidence.preview(
            project,
            path=request.path,
            start_line=request.start_line,
            end_line=request.end_line,
            reason=request.reason,
        )

    @app.post("/api/projects/{project_id}/code-evidence")
    async def add_code_evidence(project_id: str, request: CodeEvidenceRequest) -> Any:
        project = await require_project(project_id)
        evidence = code_evidence.preview(
            project,
            path=request.path,
            start_line=request.start_line,
            end_line=request.end_line,
            reason=request.reason,
        )
        return await artifact_catalog.add_code_evidence(project_id, evidence)

    @app.get("/api/projects/{project_id}/code-evidence")
    async def list_code_evidence(project_id: str) -> dict[str, Any]:
        await require_project(project_id)
        return {"codeEvidence": await artifact_catalog.list_code_evidence(project_id)}

    @app.get("/api/projects/{project_id}/evidence-preview")
    async def preview_evidence_boundary(project_id: str) -> Any:
        return await evidence_boundary.preview(await require_project(project_id))

    @app.get("/api/workflows")
    async def list_workflows(project_id: str | None = None) -> dict[str, Any]:
        return {"workflows": await active_workflow_store.list_workflows(project_id)}

    @app.post("/api/workflows", status_code=status.HTTP_201_CREATED)
    async def create_workflow(request: WorkflowCreateRequest) -> Any:
        return await active_workflow_store.create_workflow(
            project_id=request.project_id,
            name=request.name,
            description=request.description,
            idempotency_key=request.idempotency_key,
            draft=request.draft,
        )

    @app.get("/api/workflows/{workflow_id}")
    async def get_workflow(workflow_id: str) -> Any:
        workflow = await active_workflow_store.get_workflow(workflow_id)
        if workflow is None:
            raise _http_error("WORKFLOW_NOT_FOUND", "workflow does not exist", 404)
        return workflow

    @app.post("/api/workflows/{workflow_id}/resume")
    async def resume_workflow(workflow_id: str) -> Any:
        return await workflows.resume(workflow_id)

    @app.post("/api/workflows/{workflow_id}/cancel")
    async def cancel_workflow(workflow_id: str) -> Any:
        return await workflows.cancel(workflow_id)

    @app.get("/api/workflows/{workflow_id}/events")
    async def list_workflow_events(
        workflow_id: str,
        after: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        events = await active_workflow_store.list_events(
            workflow_id,
            after_revision=after,
        )
        return {
            "events": events,
            "revision": after if not events else events[-1].revision,
        }

    @app.get("/api/runs")
    async def list_runs(project_id: str = "default") -> dict[str, Any]:
        return {
            "runs": await result_store.list_runs(project_id),
            "jobs": await jobs.list(project_id),
        }

    @app.get("/api/ai/evidence")
    async def ai_evidence() -> dict[str, Any]:
        paths: list[str] = []
        for project in await active_workflow_store.list_projects():
            for artifact_type in ("GenerationReport", "AnalysisReport"):
                paths.extend(
                    artifact.path
                    for artifact in await artifact_catalog.list_artifacts(
                        project.project_id, artifact_type
                    )
                    if artifact.availability == "AVAILABLE"
                )
        return list_ai_evidence(settings.workspace_root, artifact_paths=paths)

    @app.post("/api/runs", response_model=RunJob, status_code=status.HTTP_202_ACCEPTED)
    async def create_run(request: RunCreateRequest) -> RunJob:
        try:
            return await jobs.submit(request)
        except ValueError as exc:
            raise _http_error("RUN_REQUEST_PATH_ERROR", str(exc), 422) from None

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        stored = await result_store.get_run(run_id)
        if stored is not None:
            return {"state": "stored", "storedRun": stored}
        job = await jobs.get(run_id)
        if job is not None:
            return {"state": "job", "job": job}
        raise _http_error("RUN_NOT_FOUND", "run does not exist", 404)

    @app.get("/api/runs/{run_id}/results")
    async def get_results(run_id: str) -> dict[str, Any]:
        stored = await result_store.get_run(run_id)
        if stored is None:
            raise _http_error("RUN_NOT_FOUND", "run does not exist", 404)
        return {"results": await result_store.list_results(run_id)}

    @app.get("/api/runs/{run_id}/findings")
    async def get_findings(run_id: str) -> dict[str, Any]:
        stored = await result_store.get_run(run_id)
        if stored is None:
            raise _http_error("RUN_NOT_FOUND", "run does not exist", 404)
        return {"findings": await result_store.list_findings(run_id)}

    @app.put("/api/findings/{finding_id}/review")
    async def update_review(finding_id: str, request: ReviewUpdateRequest) -> FindingReview:
        review = FindingReview(
            finding_id=finding_id,
            status=request.status,
            note=request.note,
            updated_at=utc_now(),
        )
        await result_store.save_review(review)
        return review

    @app.get("/api/traces/{trace_id}")
    async def get_trace(trace_id: str) -> Any:
        trace = await result_store.get_trace(trace_id)
        if trace is not None:
            return trace
        if live_traces.get_events(trace_id):
            return live_traces.build_record(trace_id)
        raise _http_error("TRACE_NOT_FOUND", "trace does not exist", 404)

    async def _stored_artifact(run_id: str) -> Path:
        stored = await result_store.get_run(run_id)
        if stored is None:
            raise _http_error("RUN_NOT_FOUND", "run does not exist", 404)
        if stored.artifact_path is None:
            raise _http_error(
                "RUN_ARTIFACT_UNAVAILABLE",
                "run has no artifact directory for comparison",
                409,
            )
        return Path(stored.artifact_path)

    @app.post("/api/comparisons")
    async def compare(request: ComparisonRequest) -> Any:
        try:
            return compare_runs(
                load_run_artifact(await _stored_artifact(request.baseline_run_id)),
                load_run_artifact(await _stored_artifact(request.candidate_run_id)),
            )
        except ComparisonError as exc:
            raise _http_error(exc.code, exc.message, 409) from None

    @app.post("/api/gates/evaluate")
    async def gate(request: GateEvaluateRequest) -> Any:
        try:
            gate_path = jobs.resolve_workspace_path(
                request.gate_path,
                file_okay=True,
                directory_okay=False,
            )
            loaded = load_spec_file(gate_path)
            gate_specs = [item.spec for item in loaded if isinstance(item.spec, QualityGateSpec)]
            if len(loaded) != 1 or len(gate_specs) != 1:
                raise GateError(
                    "gate file must contain exactly one QualityGate document",
                    code="GATE_SPEC_ERROR",
                )
            candidate = load_run_artifact(await _stored_artifact(request.candidate_run_id))
            comparison = None
            if request.baseline_run_id is not None:
                comparison = compare_runs(
                    load_run_artifact(await _stored_artifact(request.baseline_run_id)),
                    candidate,
                )
            return evaluate_gate(gate_specs[0], candidate, comparison)
        except SpecLoadError as exc:
            message = exc.issues[0].format() if exc.issues else exc.message
            raise _http_error("GATE_SPEC_ERROR", message, 422) from None
        except (ComparisonError, GateError) as exc:
            raise _http_error(exc.code, exc.message, 409) from None
        except ValueError as exc:
            raise _http_error("GATE_PATH_ERROR", str(exc), 422) from None

    @app.post("/v1/traces")
    async def ingest_traces(request: Request) -> Response:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type != "application/x-protobuf":
            return Response(
                b"expected application/x-protobuf\n",
                status_code=415,
                media_type="text/plain",
            )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > settings.max_receiver_bytes:
                    return Response(b"payload too large\n", status_code=413)
            except ValueError:
                return Response(b"invalid content-length\n", status_code=400)
        payload = await request.body()
        if request.headers.get("content-encoding", "").lower() == "gzip":
            try:
                payload = gzip.decompress(payload)
            except (gzip.BadGzipFile, OSError):
                return Response(b"invalid gzip payload\n", status_code=400)
        if len(payload) > settings.max_receiver_bytes:
            return Response(b"payload too large\n", status_code=413)
        try:
            response = ingest_otlp_protobuf(payload, live_traces, trace_options)
        except OTLPDecodeError:
            return Response(b"malformed protobuf\n", status_code=400)
        except Exception:
            return Response(b"trace ingestion failed\n", status_code=500)
        return Response(response, media_type="application/x-protobuf")

    static_directory = settings.static_directory
    if static_directory is not None and static_directory.is_dir():
        app.mount("/", StaticFiles(directory=static_directory, html=True), name="web")

    return app
