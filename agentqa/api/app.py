"""FastAPI composition root and HTTP routes."""

import gzip
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agentqa.api.jobs import RunJobManager
from agentqa.api.models import (
    ComparisonRequest,
    GateEvaluateRequest,
    ReviewUpdateRequest,
    RunCreateRequest,
    RunJob,
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


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Local API paths and safety limits."""

    workspace_root: Path
    artifact_directory: Path
    database_path: Path
    static_directory: Path | None = None
    max_concurrent_runs: int = 2
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
    judge_provider: JudgeProviderLike | None = None,
) -> FastAPI:
    """Build an injectable local API without global database connections."""
    settings = settings or ApiSettings.defaults()
    load_judge_environment(settings.workspace_root / ".env.local")
    active_judge = judge_provider or configured_deepseek_provider()
    result_store = store or SQLiteResultStore(settings.database_path)
    trace_options = TraceOptions(max_receiver_bytes=settings.max_receiver_bytes)
    live_traces = InMemoryTraceStore(max_spans_per_trace=trace_options.max_spans_per_trace)
    jobs = RunJobManager(
        result_store,
        workspace_root=settings.workspace_root,
        artifact_directory=settings.artifact_directory,
        judge_provider=active_judge,
        max_concurrent_runs=settings.max_concurrent_runs,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        await result_store.initialize()
        try:
            yield
        finally:
            await jobs.shutdown()
            await result_store.close()

    app = FastAPI(
        title="AgentQAKit API",
        version="v1alpha1",
        lifespan=lifespan,
    )
    app.state.result_store = result_store
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

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "storage": type(result_store).__name__,
            "workspaceRoot": str(settings.workspace_root.resolve()),
            "judge": {
                "configured": active_judge is not None,
                "provider": None if active_judge is None else active_judge.name,
                "model": None if active_judge is None else active_judge.model,
            },
        }

    @app.get("/api/runs")
    async def list_runs(project_id: str = "default") -> dict[str, Any]:
        return {
            "runs": await result_store.list_runs(project_id),
            "jobs": await jobs.list(project_id),
        }

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
