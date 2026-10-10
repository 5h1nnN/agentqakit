"""In-process background execution registry for the local API."""

import asyncio
from pathlib import Path

from agentqa.api.models import RunCreateRequest, RunJob
from agentqa.application import RunExecutionConfig, execute_run
from agentqa.conventions import new_id, utc_now
from agentqa.errors import AgentQAError
from agentqa.evaluators import JudgeProvider
from agentqa.models import ErrorInfo
from agentqa.specs import SpecLoadError
from agentqa.storage import ResultStore


class RunJobManager:
    """Own bounded local jobs without introducing an external queue."""

    def __init__(
        self,
        store: ResultStore,
        *,
        workspace_root: Path,
        artifact_directory: Path,
        judge_provider: JudgeProvider | None = None,
        max_concurrent_runs: int = 2,
    ) -> None:
        self._store = store
        self._workspace_root = workspace_root.resolve()
        self._artifact_directory = artifact_directory.resolve()
        self._judge_provider = judge_provider
        self._semaphore = asyncio.Semaphore(max_concurrent_runs)
        self._jobs: dict[str, RunJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def resolve_workspace_path(
        self,
        value: str,
        *,
        file_okay: bool = True,
        directory_okay: bool = True,
    ) -> Path:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("paths must be workspace-relative and cannot contain '..'")
        resolved = (self._workspace_root / relative).resolve()
        try:
            resolved.relative_to(self._workspace_root)
        except ValueError:
            raise ValueError("path escapes the configured workspace") from None
        if not resolved.exists():
            raise ValueError("workspace path does not exist")
        if resolved.is_file() and not file_okay:
            raise ValueError("workspace path must be a directory")
        if resolved.is_dir() and not directory_okay:
            raise ValueError("workspace path must be a file")
        return resolved

    async def submit(self, request: RunCreateRequest) -> RunJob:
        if request.judge_enabled and self._judge_provider is None:
            raise ValueError("DeepSeek Judge is not configured on this server")
        tests_path = self.resolve_workspace_path(request.tests_path)
        target_path = (
            None
            if request.target_path is None
            else self.resolve_workspace_path(request.target_path)
        )
        mutations_path = (
            None
            if request.mutations_path is None
            else self.resolve_workspace_path(request.mutations_path)
        )
        run_id = new_id()
        job = RunJob(
            run_id=run_id,
            project_id=request.project_id,
            status="QUEUED",
            submitted_at=utc_now(),
        )
        async with self._lock:
            self._jobs[run_id] = job
            self._tasks[run_id] = asyncio.create_task(
                self._execute(
                    job,
                    request,
                    tests_path,
                    target_path,
                    mutations_path,
                ),
                name=f"agentqa-run-{run_id}",
            )
        return job

    async def _set_job(self, job: RunJob) -> None:
        async with self._lock:
            self._jobs[job.run_id] = job

    async def _execute(
        self,
        job: RunJob,
        request: RunCreateRequest,
        tests_path: Path,
        target_path: Path | None,
        mutations_path: Path | None,
    ) -> None:
        async with self._semaphore:
            await self._set_job(job.model_copy(update={"status": "RUNNING"}))
            try:
                result = await execute_run(
                    RunExecutionConfig(
                        tests_path=tests_path,
                        target_path=target_path,
                        mutations_path=mutations_path,
                        artifact_directory=self._artifact_directory,
                        random_seed=request.random_seed,
                        concurrency=request.concurrency,
                        max_total_tokens=request.max_total_tokens,
                        max_total_cost_usd=request.max_total_cost_usd,
                        trace_receiver=False,
                        capture_trace_content=request.capture_trace_content,
                        generate_failure_bundles=request.generate_failure_bundles,
                        original_command="agentqa API background run",
                        run_id=job.run_id,
                        judge_provider=(self._judge_provider if request.judge_enabled else None),
                    )
                )
                artifact_path = self._artifact_directory / "runs" / result.run_id
                await self._store.save_run(
                    result,
                    project_id=job.project_id,
                    artifact_path=artifact_path,
                )
                await self._set_job(
                    job.model_copy(
                        update={
                            "status": result.status,
                            "finished_at": utc_now(),
                            "error": result.error,
                        }
                    )
                )
            except asyncio.CancelledError:
                await self._set_job(
                    job.model_copy(update={"status": "CANCELLED", "finished_at": utc_now()})
                )
                raise
            except SpecLoadError as exc:
                message = exc.issues[0].format() if exc.issues else exc.message
                await self._fail(job, "SPEC_ERROR", message)
            except AgentQAError as exc:
                await self._fail(job, exc.code, exc.message)
            except Exception as exc:
                await self._fail(
                    job,
                    "RUN_SERVICE_ERROR",
                    f"background run raised {type(exc).__name__}",
                )

    async def _fail(self, job: RunJob, code: str, message: str) -> None:
        await self._set_job(
            job.model_copy(
                update={
                    "status": "FAILED",
                    "finished_at": utc_now(),
                    "error": ErrorInfo(code=code, message=message),
                }
            )
        )

    async def get(self, run_id: str) -> RunJob | None:
        async with self._lock:
            return self._jobs.get(run_id)

    async def list(self, project_id: str) -> list[RunJob]:
        async with self._lock:
            jobs = [job for job in self._jobs.values() if job.project_id == project_id]
        return sorted(jobs, key=lambda item: item.submitted_at, reverse=True)

    async def wait(self, run_id: str) -> RunJob | None:
        async with self._lock:
            task = self._tasks.get(run_id)
        if task is not None:
            await task
        return await self.get(run_id)

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = list(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
