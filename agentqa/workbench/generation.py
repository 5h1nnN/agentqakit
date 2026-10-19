"""Two-phase AI generation and human review for the local Web workbench."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import JsonValue, ValidationError

from agentqa.application import GenerationError, GenerationService
from agentqa.conventions import new_id, utc_now
from agentqa.errors import AgentQAError
from agentqa.generation import (
    load_generation_plan,
    write_generation_plan,
    write_generation_suite,
)
from agentqa.llm import StructuredLLMProvider
from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    ErrorInfo,
    GenerationGoal,
    GenerationGoalSummary,
    GenerationReport,
    MutationSpec,
)
from agentqa.specs import SpecLoadError, hash_spec, load_spec_file, load_specs
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.storage import redact_text_secrets
from agentqa.workbench.catalog import ArtifactCatalogService
from agentqa.workbench.generation_store import (
    GenerationWorkbenchError,
    GenerationWorkbenchStore,
)
from agentqa.workbench.models import (
    ApprovalDecision,
    ArtifactRef,
    CandidateDiagnostic,
    GeneratedCandidateView,
    GeneratedTestRevision,
    GenerationSession,
    SimilarityHint,
    WorkbenchProject,
    WorkflowStep,
)
from agentqa.workbench.orchestrator import WorkflowOrchestrator
from agentqa.workbench.store import WorkflowStore

_SUPPORTED_EVALUATORS = frozenset(
    {"schema", "text", "tool_call", "trajectory", "state", "budget", "privacy", "llm_judge"}
)


class GenerationWorkspaceError(AgentQAError):
    def __init__(self, message: str, *, code: str = "GENERATION_WORKSPACE_ERROR") -> None:
        super().__init__(message, code=code)


def _single_spec(path: Path, expected: type[Any]) -> Any:
    try:
        matches = [item.spec for item in load_spec_file(path) if isinstance(item.spec, expected)]
    except (OSError, SpecLoadError):
        raise GenerationWorkspaceError(
            f"{path.name} could not be loaded as {expected.__name__}",
            code="GENERATION_SPEC_ERROR",
        ) from None
    if len(matches) != 1:
        raise GenerationWorkspaceError(
            f"{path.name} must contain exactly one {expected.__name__}",
            code="GENERATION_SPEC_SELECTION_ERROR",
        )
    return matches[0]


def _safe_project_path(workspace_root: Path, project: WorkbenchProject, raw: str) -> Path:
    project_root = (workspace_root / PurePosixPath(project.workspace_path)).resolve()
    candidate = (project_root / PurePosixPath(raw)).resolve()
    try:
        project_root.relative_to(workspace_root.resolve())
        candidate.relative_to(project_root)
    except ValueError:
        raise GenerationWorkspaceError(
            "generation path escapes the selected project",
            code="GENERATION_PATH_ERROR",
        ) from None
    return candidate


def _summary(goal: GenerationGoal) -> GenerationGoalSummary:
    return GenerationGoalSummary(
        categories=goal.categories,
        risk_levels=goal.risk_levels,
        target_capabilities=goal.target_capabilities,
        target_policies=goal.target_policies,
        allowed_mutations=goal.allowed_mutations,
        max_cases=goal.max_cases,
        instructions=goal.instructions,
        code_evidence_hashes=[item.content_hash for item in goal.code_evidence],
    )


def _tokens(call: Any) -> int:
    usage = call.token_usage
    total = usage.get("total_tokens")
    if total is not None:
        return int(total)
    return int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))


def _artifact_ref(artifact: Any) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
        uri=artifact.path,
        schema_version=artifact.schema_version,
        checksum=artifact.checksum,
        source="workflow",
    )


def _load_report(path: Path) -> GenerationReport:
    try:
        payload = path.read_bytes()
        expected = path.with_name(f"{path.name}.sha256").read_text(encoding="utf-8").strip()
    except OSError:
        raise GenerationWorkspaceError(
            "generation report or checksum could not be read",
            code="GENERATION_REPORT_READ_ERROR",
        ) from None
    if expected != sha256_digest(payload):
        raise GenerationWorkspaceError(
            "generation report checksum mismatch",
            code="GENERATION_REPORT_CHECKSUM_ERROR",
        )
    try:
        return GenerationReport.model_validate_json(payload)
    except ValidationError:
        raise GenerationWorkspaceError(
            "generation report does not match its schema",
            code="GENERATION_REPORT_SCHEMA_ERROR",
        ) from None


def _tool_references(spec: AgentTestSpec) -> set[str]:
    tools = spec.expect.tools
    if tools is None:
        return set()
    references = set(tools.required) | set(tools.forbidden) | set(tools.argument_schemas)
    if isinstance(tools.min_calls, dict):
        references.update(tools.min_calls)
    if isinstance(tools.max_calls, dict):
        references.update(tools.max_calls)
    for order in tools.order:
        references.update((order.before, order.after))
    return references


def _contains_template(value: JsonValue) -> bool:
    if isinstance(value, str):
        return "{{" in value or "{%" in value
    if isinstance(value, list):
        return any(_contains_template(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_template(item) for item in value.values())
    return False


def _validate_edited_spec(
    spec: AgentTestSpec,
    manifest: AgentManifest,
    allowed_mutations: set[str],
) -> list[CandidateDiagnostic]:
    diagnostics: list[CandidateDiagnostic] = []
    allowed_tools = {item.name for item in manifest.tools}
    unknown_tools = sorted(_tool_references(spec) - allowed_tools)
    if spec.target_ref is not None:
        diagnostics.append(
            CandidateDiagnostic(
                code="TARGET_REF_NOT_ALLOWED",
                status="BLOCK",
                message="Generated tests must omit targetRef until campaign setup.",
            )
        )
    if unknown_tools:
        diagnostics.append(
            CandidateDiagnostic(
                code="UNDECLARED_TOOL",
                status="BLOCK",
                message="The edit references a tool not declared by the Manifest.",
            )
        )
    evaluator_types = {item.type for item in spec.evaluators}
    if evaluator_types - _SUPPORTED_EVALUATORS:
        diagnostics.append(
            CandidateDiagnostic(
                code="UNSUPPORTED_EVALUATOR",
                status="BLOCK",
                message="The edit contains an unsupported evaluator.",
            )
        )
    if evaluator_types == {"llm_judge"}:
        diagnostics.append(
            CandidateDiagnostic(
                code="NONDETERMINISTIC_ONLY",
                status="BLOCK",
                message="llm_judge must supplement a deterministic evaluator.",
            )
        )
    if spec.mutation is not None and spec.mutation not in allowed_mutations:
        diagnostics.append(
            CandidateDiagnostic(
                code="UNAVAILABLE_MUTATION",
                status="BLOCK",
                message="The edit references a Mutation outside the approved goal.",
            )
        )
    dumped = cast(JsonValue, spec.model_dump(mode="json", by_alias=True))
    if _contains_template(dumped):
        diagnostics.append(
            CandidateDiagnostic(
                code="TEMPLATE_NOT_ALLOWED",
                status="BLOCK",
                message="Generated tests cannot contain template expressions.",
            )
        )
    if not diagnostics:
        diagnostics.append(
            CandidateDiagnostic(
                code="STATIC_VALIDATION_PASS",
                status="PASS",
                message="Schema, tool, evaluator, mutation, and template checks passed.",
            )
        )
    return diagnostics


def _similarity(left: AgentTestSpec, right: AgentTestSpec) -> float:
    def words(value: AgentTestSpec) -> set[str]:
        payload = json.dumps(
            {
                "scenario": value.scenario.model_dump(mode="json", by_alias=True),
                "expect": value.expect.model_dump(mode="json", by_alias=True),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).lower()
        return set(re.findall(r"[\w\u4e00-\u9fff]+", payload))

    first, second = words(left), words(right)
    return 0.0 if not first and not second else len(first & second) / max(1, len(first | second))


def _semantic_hash(spec: AgentTestSpec) -> str:
    return sha256_digest(
        canonical_json_bytes(
            {
                "scenario": spec.scenario.model_dump(mode="json", by_alias=True),
                "expect": spec.expect.model_dump(mode="json", by_alias=True),
                "mutation": spec.mutation,
            }
        )
    )


class GenerationWorkbenchService:
    """Coordinate persistent sessions, immutable artifacts, and checksum-bound reviews."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        artifact_directory: Path,
        provider: StructuredLLMProvider | None,
        workflow_store: WorkflowStore,
        session_store: GenerationWorkbenchStore,
        artifact_catalog: ArtifactCatalogService,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.artifact_directory = artifact_directory.resolve()
        self.provider = provider
        self.workflow_store = workflow_store
        self.session_store = session_store
        self.artifact_catalog = artifact_catalog
        self.orchestrator = orchestrator

    async def _project_and_workflow(self, workflow_id: str) -> tuple[WorkbenchProject, Any]:
        workflow = await self.workflow_store.get_workflow(workflow_id)
        if workflow is None:
            raise GenerationWorkspaceError("workflow does not exist", code="WORKFLOW_NOT_FOUND")
        project = await self.workflow_store.get_project(workflow.project_id)
        if project is None:
            raise GenerationWorkspaceError(
                "project does not exist", code="WORKBENCH_PROJECT_NOT_FOUND"
            )
        return project, workflow

    async def create_and_plan(
        self,
        *,
        workflow_id: str,
        idempotency_key: str,
        goal: GenerationGoal | None,
        goal_path: str | None,
        manifest_path: str | None,
        mutations_path: str | None,
        evidence_mode: str,
        code_evidence_hashes: list[str],
        max_output_tokens: int,
        max_total_tokens: int | None,
        max_cost_usd: float | None,
    ) -> tuple[GenerationSession, WorkflowStep]:
        project, workflow = await self._project_and_workflow(workflow_id)
        if workflow.status != "READY":
            raise GenerationWorkspaceError(
                "workflow must be READY before generation starts",
                code="WORKFLOW_TRANSITION_ERROR",
            )
        manifest_ref = manifest_path or project.default_spec_refs.get("manifest")
        if manifest_ref is None:
            raise GenerationWorkspaceError(
                "project has no Manifest reference", code="GENERATION_MANIFEST_MISSING"
            )
        if goal is not None and goal_path is not None:
            raise GenerationWorkspaceError(
                "choose either a saved GenerationGoal or a new inline goal",
                code="GENERATION_GOAL_CONFLICT",
            )
        goal_ref = goal_path or (
            None if goal is not None else project.default_spec_refs.get("generation_goal")
        )
        if goal is None:
            if goal_ref is None:
                goal = GenerationGoal.model_validate({"metadata": {"name": "web-generation"}})
            else:
                goal = cast(
                    GenerationGoal,
                    _single_spec(
                        _safe_project_path(self.workspace_root, project, goal_ref),
                        GenerationGoal,
                    ),
                )
        selected = await self.artifact_catalog.list_code_evidence(project.project_id)
        selected_by_hash = {item.content_hash: item for item in selected}
        if len(code_evidence_hashes) != len(set(code_evidence_hashes)):
            raise GenerationWorkspaceError(
                "code evidence hashes must be unique", code="GENERATION_EVIDENCE_ERROR"
            )
        if evidence_mode == "MANIFEST_ONLY":
            if code_evidence_hashes:
                raise GenerationWorkspaceError(
                    "manifest-only mode cannot include code evidence",
                    code="GENERATION_EVIDENCE_ERROR",
                )
            evidence = []
        elif evidence_mode == "MANIFEST_AND_CODE":
            missing = set(code_evidence_hashes) - set(selected_by_hash)
            if missing or not code_evidence_hashes:
                raise GenerationWorkspaceError(
                    "code-assisted mode requires reviewed CodeEvidence selections",
                    code="GENERATION_EVIDENCE_ERROR",
                )
            evidence = [selected_by_hash[item] for item in code_evidence_hashes]
        else:
            raise GenerationWorkspaceError(
                "unsupported generation evidence mode", code="GENERATION_EVIDENCE_ERROR"
            )
        controlled_goal = goal.model_copy(update={"code_evidence": evidence})
        material = {
            "workflow_id": workflow_id,
            "goal_hash": hash_spec(controlled_goal),
            "manifest_path": manifest_ref,
            "mutations_path": mutations_path or project.default_spec_refs.get("mutations"),
            "evidence_mode": evidence_mode,
            "max_output_tokens": max_output_tokens,
            "max_total_tokens": max_total_tokens,
            "max_cost_usd": max_cost_usd,
        }
        input_hash = sha256_digest(canonical_json_bytes(material))
        now = utc_now()
        proposed = GenerationSession(
            session_id=new_id(),
            workflow_id=workflow_id,
            project_id=project.project_id,
            status="PLANNING",
            idempotency_key=idempotency_key,
            input_hash=input_hash,
            goal=_summary(controlled_goal),
            manifest_path=manifest_ref,
            mutations_path=cast(str | None, material["mutations_path"]),
            evidence_mode=cast(Any, evidence_mode),
            code_evidence_hashes=code_evidence_hashes,
            max_output_tokens=max_output_tokens,
            max_total_tokens=max_total_tokens,
            max_cost_usd=max_cost_usd,
            created_at=now,
            updated_at=now,
        )
        session = await self.session_store.create_session(proposed)
        if session.session_id == proposed.session_id:
            await self.workflow_store.record_event(
                workflow_id,
                event_type="GENERATION_SESSION_CREATED",
                message="Created a two-phase AI generation session.",
                data={
                    "generation_session_id": session.session_id,
                    "input_hash": session.input_hash,
                    "evidence_mode": session.evidence_mode,
                },
            )

        async def handler(step: WorkflowStep) -> list[ArtifactRef]:
            del step
            return await self._generate_plan(session.session_id, controlled_goal)

        workflow_step = await self.orchestrator.submit(
            workflow_id=workflow_id,
            kind="GENERATE_PLAN",
            idempotency_key=f"generation-plan:{idempotency_key}",
            input_data=cast(
                dict[str, JsonValue],
                {
                    "generation_session_id": session.session_id,
                    "goal_hash": hash_spec(controlled_goal),
                    "manifest_path": manifest_ref,
                    "evidence_mode": evidence_mode,
                    "code_evidence_hashes": code_evidence_hashes,
                    "max_output_tokens": max_output_tokens,
                    "max_total_tokens": max_total_tokens,
                    "max_cost_usd": max_cost_usd,
                },
            ),
            handler=handler,
        )
        return session, workflow_step

    async def _fail_session(self, session: GenerationSession, exc: AgentQAError) -> None:
        calls = list(session.llm_calls)
        if isinstance(exc, GenerationError) and exc.llm_call is not None:
            calls.append(exc.llm_call)
        failed = session.model_copy(
            update={
                "status": "FAILED_SAFE",
                "llm_calls": calls[-2:],
                "error": ErrorInfo(code=exc.code, message=redact_text_secrets(exc.message)),
                "updated_at": utc_now(),
            }
        )
        await self.session_store.update_session(failed)

    def _check_budget(self, session: GenerationSession, calls: list[Any]) -> int:
        consumed = sum(_tokens(item) for item in calls)
        if session.max_total_tokens is not None and consumed > session.max_total_tokens:
            raise GenerationWorkspaceError(
                "generation exceeded the configured token budget",
                code="GENERATION_TOKEN_BUDGET_EXCEEDED",
            )
        if session.max_cost_usd is not None:
            raise GenerationWorkspaceError(
                "provider did not report observable generation cost",
                code="GENERATION_COST_UNOBSERVABLE",
            )
        return consumed

    async def _generate_plan(self, session_id: str, goal: GenerationGoal) -> list[ArtifactRef]:
        session = await self.require_session(session_id)
        try:
            if self.provider is None:
                raise GenerationWorkspaceError(
                    "structured LLM provider is not configured",
                    code="GENERATION_PROVIDER_NOT_CONFIGURED",
                )
            project = await self.workflow_store.get_project(session.project_id)
            if project is None:
                raise GenerationWorkspaceError("project does not exist")
            manifest = cast(
                AgentManifest,
                _single_spec(
                    _safe_project_path(self.workspace_root, project, session.manifest_path),
                    AgentManifest,
                ),
            )
            plan = await GenerationService(
                self.provider, max_output_tokens=session.max_output_tokens
            ).generate_plan(manifest, goal)
            session = session.model_copy(
                update={
                    "llm_calls": [plan.llm_call],
                    "consumed_tokens": _tokens(plan.llm_call),
                }
            )
            consumed = self._check_budget(session, [plan.llm_call])
            output = (
                self.artifact_directory
                / "generations"
                / session.session_id
                / "generation-plan.json"
            )
            write_generation_plan(plan, output, workspace=self.workspace_root)
            artifact = await self.artifact_catalog.register_path(
                session.project_id,
                output,
                source="WORKFLOW",
                workflow_id=session.workflow_id,
            )
            updated = session.model_copy(
                update={
                    "status": "PLAN_READY",
                    "plan_path": artifact.path,
                    "plan_checksum": artifact.checksum,
                    "consumed_tokens": consumed,
                    "llm_calls": [plan.llm_call],
                    "updated_at": utc_now(),
                }
            )
            await self.session_store.update_session(updated)
            return [_artifact_ref(artifact)]
        except AgentQAError as exc:
            await self._fail_session(session, exc)
            raise
        except Exception as exc:
            safe = GenerationWorkspaceError(
                f"generation plan step raised {type(exc).__name__}",
                code="GENERATION_STEP_ERROR",
            )
            await self._fail_session(session, safe)
            raise safe from None

    async def generate_tests(
        self,
        *,
        session_id: str,
        plan_checksum: str,
        idempotency_key: str,
    ) -> tuple[GenerationSession, WorkflowStep]:
        session = await self.require_session(session_id)
        if session.status != "PLAN_READY" or session.plan_checksum != plan_checksum:
            raise GenerationWorkspaceError(
                "candidate generation requires confirmation of the current verified plan",
                code="GENERATION_PLAN_CONFIRMATION_ERROR",
            )
        # The plan artifact becomes visible just before the orchestrator marks
        # its workflow step complete. Absorb that brief, normal transition.
        workflow = await self.workflow_store.get_workflow(session.workflow_id)
        if workflow is not None and workflow.status == "RUNNING":
            active_plan = next(
                (step for step in reversed(workflow.steps)
                 if step.kind == "GENERATE_PLAN" and step.status == "RUNNING"),
                None,
            )
            if active_plan is not None:
                try:
                    await asyncio.wait_for(self.orchestrator.wait(active_plan.step_id), timeout=5)
                except TimeoutError:
                    raise GenerationWorkspaceError(
                        "plan is still finalizing; retry candidate generation shortly",
                        code="GENERATION_PLAN_FINALIZING",
                    ) from None
        generating = session.model_copy(update={"status": "GENERATING", "updated_at": utc_now()})
        await self.session_store.update_session(generating)

        async def handler(step: WorkflowStep) -> list[ArtifactRef]:
            del step
            return await self._generate_tests(session_id)

        step = await self.orchestrator.submit(
            workflow_id=session.workflow_id,
            kind="GENERATE_TESTS",
            idempotency_key=f"generation-tests:{idempotency_key}",
            input_data={
                "generation_session_id": session_id,
                "plan_checksum": plan_checksum,
            },
            handler=handler,
        )
        return generating, step

    async def _generate_tests(self, session_id: str) -> list[ArtifactRef]:
        session = await self.require_session(session_id)
        try:
            if self.provider is None:
                raise GenerationWorkspaceError(
                    "structured LLM provider is not configured",
                    code="GENERATION_PROVIDER_NOT_CONFIGURED",
                )
            project = await self.workflow_store.get_project(session.project_id)
            if project is None or session.plan_path is None:
                raise GenerationWorkspaceError("generation project or plan is unavailable")
            manifest = cast(
                AgentManifest,
                _single_spec(
                    _safe_project_path(self.workspace_root, project, session.manifest_path),
                    AgentManifest,
                ),
            )
            plan = load_generation_plan(self.workspace_root / PurePosixPath(session.plan_path))
            mutations: list[MutationSpec] = []
            if session.goal.allowed_mutations:
                if session.mutations_path is None:
                    raise GenerationWorkspaceError(
                        "generation goal allows mutations but no Mutation path is configured",
                        code="GENERATION_MUTATION_MISSING",
                    )
                try:
                    mutations = [
                        item.spec
                        for item in load_specs(
                            _safe_project_path(self.workspace_root, project, session.mutations_path)
                        )
                        if isinstance(item.spec, MutationSpec)
                    ]
                except (OSError, SpecLoadError):
                    raise GenerationWorkspaceError(
                        "Mutation specs could not be loaded",
                        code="GENERATION_MUTATION_ERROR",
                    ) from None
            report = await GenerationService(
                self.provider, max_output_tokens=session.max_output_tokens
            ).generate_tests(manifest, plan, mutations)
            calls = report.llm_calls
            session = session.model_copy(
                update={
                    "llm_calls": calls,
                    "consumed_tokens": sum(_tokens(item) for item in calls),
                }
            )
            consumed = self._check_budget(session, calls)
            output = self.artifact_directory / "generations" / session.session_id / "tests"
            written = write_generation_suite(report, output, workspace=self.workspace_root)
            report_path = next(path for path in written if path.name == "generation-report.json")
            report_artifact = await self.artifact_catalog.register_path(
                session.project_id,
                report_path,
                source="WORKFLOW",
                workflow_id=session.workflow_id,
            )
            refs = [_artifact_ref(report_artifact)]
            for path in written:
                if path == report_path:
                    continue
                checksum = path.with_name(f"{path.name}.sha256").read_text(encoding="utf-8").strip()
                refs.append(
                    ArtifactRef(
                        artifact_id=sha256_digest(
                            canonical_json_bytes(
                                {
                                    "path": path.relative_to(self.workspace_root).as_posix(),
                                    "checksum": checksum,
                                }
                            )
                        ),
                        artifact_type="AgentTest",
                        uri=path.relative_to(self.workspace_root).as_posix(),
                        schema_version="agentqa.dev/v1alpha1",
                        checksum=checksum,
                        source="workflow",
                    )
                )
            ready = session.model_copy(
                update={
                    "status": "CANDIDATES_READY",
                    "report_path": report_artifact.path,
                    "report_checksum": report_artifact.checksum,
                    "consumed_tokens": consumed,
                    "llm_calls": calls,
                    "updated_at": utc_now(),
                }
            )
            await self.session_store.update_session(ready)
            return refs
        except AgentQAError as exc:
            await self._fail_session(session, exc)
            raise
        except Exception as exc:
            safe = GenerationWorkspaceError(
                f"candidate generation step raised {type(exc).__name__}",
                code="GENERATION_STEP_ERROR",
            )
            await self._fail_session(session, safe)
            raise safe from None

    async def require_session(self, session_id: str) -> GenerationSession:
        session = await self.session_store.get_session(session_id)
        if session is None:
            raise GenerationWorkbenchError(
                "generation session does not exist", code="GENERATION_SESSION_NOT_FOUND"
            )
        return session

    async def detail(self, session_id: str) -> dict[str, Any]:
        session = await self.require_session(session_id)
        plan = None
        if session.plan_path is not None:
            plan = load_generation_plan(self.workspace_root / PurePosixPath(session.plan_path))
        candidates: list[GeneratedCandidateView] = []
        if session.report_path is not None:
            candidates = await self.candidates(session_id)
        return {"session": session, "plan": plan, "candidates": candidates}

    async def _context(
        self, session: GenerationSession
    ) -> tuple[WorkbenchProject, AgentManifest, GenerationReport]:
        project = await self.workflow_store.get_project(session.project_id)
        if project is None or session.report_path is None:
            raise GenerationWorkspaceError(
                "candidate report is not available", code="GENERATION_REPORT_NOT_READY"
            )
        manifest = cast(
            AgentManifest,
            _single_spec(
                _safe_project_path(self.workspace_root, project, session.manifest_path),
                AgentManifest,
            ),
        )
        report = _load_report(self.workspace_root / PurePosixPath(session.report_path))
        return project, manifest, report

    async def _historical_specs(self, project: WorkbenchProject) -> list[tuple[str, AgentTestSpec]]:
        tests_ref = project.default_spec_refs.get("tests")
        if tests_ref is None:
            return []
        try:
            return [
                (
                    item.source.relative_to(
                        _safe_project_path(self.workspace_root, project, ".")
                    ).as_posix(),
                    item.spec,
                )
                for item in load_specs(_safe_project_path(self.workspace_root, project, tests_ref))
                if isinstance(item.spec, AgentTestSpec)
            ]
        except (OSError, SpecLoadError, ValueError):
            return []

    async def candidates(self, session_id: str) -> list[GeneratedCandidateView]:
        session = await self.require_session(session_id)
        project, manifest, report = await self._context(session)
        revisions = await self.session_store.list_revisions(session_id)
        by_case: dict[str, GeneratedTestRevision] = {}
        for stored_revision in revisions:
            by_case[stored_revision.case_id] = stored_revision
        approvals = await self.workflow_store.list_approvals(session.workflow_id)
        historical = await self._historical_specs(project)
        effective_specs = {
            case.case_id: (case.spec if case.case_id not in by_case else by_case[case.case_id].spec)
            for case in report.cases
        }
        semantic_counts: dict[str, int] = {}
        for spec in effective_specs.values():
            semantic = _semantic_hash(spec)
            semantic_counts[semantic] = semantic_counts.get(semantic, 0) + 1
        views: list[GeneratedCandidateView] = []
        for case in report.cases:
            revision = by_case.get(case.case_id)
            spec = effective_specs[case.case_id]
            current_hash = hash_spec(spec)
            decision = next(
                (
                    item
                    for item in reversed(approvals)
                    if item.subject_kind == "GeneratedTestRevision"
                    and item.subject_id == f"{session_id}:{case.case_id}"
                    and item.subject_hash == current_hash
                ),
                None,
            )
            diagnostics = _validate_edited_spec(spec, manifest, set(session.goal.allowed_mutations))
            if semantic_counts[_semantic_hash(spec)] > 1:
                diagnostics = [
                    CandidateDiagnostic(
                        code="DUPLICATE_CANDIDATE",
                        status="BLOCK",
                        message="Scenario, expectations, and mutation duplicate another candidate.",
                    )
                ]
            hints = sorted(
                (
                    SimilarityHint(source=source, score=round(_similarity(spec, other), 4))
                    for source, other in historical
                    if other.metadata.name != case.case_id
                ),
                key=lambda item: item.score,
                reverse=True,
            )[:3]
            blocked = any(item.status == "BLOCK" for item in diagnostics)
            rejected = decision is not None and decision.decision != "APPROVED"
            approved_when_required = case.risk not in {"high", "critical"} or (
                decision is not None and decision.decision == "APPROVED"
            )
            views.append(
                GeneratedCandidateView(
                    case_id=case.case_id,
                    risk=case.risk,
                    review_status=(
                        decision.decision
                        if decision is not None
                        else ("REVIEW_REQUIRED" if case.risk in {"high", "critical"} else "READY")
                    ),
                    rationale=case.rationale,
                    assertion_basis=case.assertion_basis,
                    coverage_item_ids=case.coverage_item_ids,
                    source_refs=case.source_refs,
                    original_spec_hash=hash_spec(case.spec),
                    current_spec_hash=current_hash,
                    spec=spec,
                    latest_revision=revision,
                    decision=decision,
                    diagnostics=diagnostics,
                    similarity_hints=[item for item in hints if item.score >= 0.35],
                    campaign_eligible=not blocked and not rejected and approved_when_required,
                )
            )
        return views

    async def revise(
        self,
        *,
        session_id: str,
        case_id: str,
        spec: AgentTestSpec,
        editor: str,
        diff_summary: str,
    ) -> GeneratedTestRevision:
        session = await self.require_session(session_id)
        _, manifest, report = await self._context(session)
        case = next((item for item in report.cases if item.case_id == case_id), None)
        if case is None:
            raise GenerationWorkspaceError(
                "generated candidate does not exist", code="GENERATION_CASE_NOT_FOUND"
            )
        if spec.metadata.name != case_id:
            raise GenerationWorkspaceError(
                "edited spec must retain the generated case name",
                code="GENERATION_REVISION_IDENTITY_ERROR",
            )
        if (
            spec.metadata.labels.get("risk") != case.risk
            or spec.metadata.labels.get("generated") != "true"
            or spec.metadata.labels.get("source") != "ai-generated"
        ):
            raise GenerationWorkspaceError(
                "generated identity and risk labels cannot be changed by a revision",
                code="GENERATION_REVISION_IDENTITY_ERROR",
            )
        diagnostics = _validate_edited_spec(spec, manifest, set(session.goal.allowed_mutations))
        prior_by_case: dict[str, GeneratedTestRevision] = {}
        for stored_revision in await self.session_store.list_revisions(session_id):
            prior_by_case[stored_revision.case_id] = stored_revision
        for other in report.cases:
            if other.case_id == case_id:
                continue
            other_revision = prior_by_case.get(other.case_id)
            other_spec = other.spec if other_revision is None else other_revision.spec
            if _semantic_hash(spec) == _semantic_hash(other_spec):
                diagnostics = [
                    CandidateDiagnostic(
                        code="DUPLICATE_CANDIDATE",
                        status="BLOCK",
                        message="The edit duplicates another generated candidate.",
                    )
                ]
                break
        if any(item.status == "BLOCK" for item in diagnostics):
            raise GenerationWorkspaceError(
                diagnostics[0].message, code="GENERATION_REVISION_VALIDATION_ERROR"
            )
        prior = await self.session_store.list_revisions(session_id, case_id)
        parent_hash = hash_spec(case.spec) if not prior else prior[-1].spec_hash
        revision = GeneratedTestRevision(
            revision_id=new_id(),
            session_id=session_id,
            case_id=case_id,
            revision=len(prior) + 1,
            parent_hash=parent_hash,
            spec_hash=hash_spec(spec),
            spec=spec,
            editor=editor,
            diff_summary=diff_summary,
            created_at=utc_now(),
        )
        saved = await self.session_store.save_revision(revision)
        if saved.revision_id == revision.revision_id:
            await self.workflow_store.record_event(
                session.workflow_id,
                event_type="GENERATED_TEST_REVISED",
                message="Created an immutable human revision of a generated test.",
                data={
                    "generation_session_id": session_id,
                    "case_id": case_id,
                    "revision_id": saved.revision_id,
                    "parent_hash": saved.parent_hash,
                    "spec_hash": saved.spec_hash,
                },
            )
        return saved

    async def review(
        self,
        *,
        session_id: str,
        case_id: str,
        subject_hash: str,
        decision: str,
        reviewer: str,
        comment: str,
    ) -> ApprovalDecision:
        session = await self.require_session(session_id)
        candidates = await self.candidates(session_id)
        case = next((item for item in candidates if item.case_id == case_id), None)
        if case is None:
            raise GenerationWorkspaceError(
                "generated candidate does not exist", code="GENERATION_CASE_NOT_FOUND"
            )
        if case.current_spec_hash != subject_hash:
            raise GenerationWorkspaceError(
                "review hash is stale; refresh the current revision",
                code="GENERATION_REVIEW_HASH_MISMATCH",
            )
        if decision == "APPROVED" and any(item.status == "BLOCK" for item in case.diagnostics):
            raise GenerationWorkspaceError(
                "a candidate with blocking diagnostics cannot be approved",
                code="GENERATION_REVIEW_BLOCKED",
            )
        approval = ApprovalDecision(
            approval_id=new_id(),
            workflow_id=session.workflow_id,
            subject_kind="GeneratedTestRevision",
            subject_id=f"{session_id}:{case_id}",
            subject_hash=subject_hash,
            decision=cast(Any, decision),
            reviewer=reviewer,
            comment=comment,
            created_at=utc_now(),
        )
        return await self.workflow_store.save_approval(approval)

    async def batch_approve(
        self, *, session_id: str, case_ids: list[str], reviewer: str, comment: str
    ) -> list[ApprovalDecision]:
        candidates = {item.case_id: item for item in await self.candidates(session_id)}
        if len(case_ids) != len(set(case_ids)):
            raise GenerationWorkspaceError(
                "batch candidate ids must be unique",
                code="GENERATION_BATCH_REVIEW_ERROR",
            )
        for case_id in case_ids:
            case = candidates.get(case_id)
            if case is None:
                raise GenerationWorkspaceError(
                    "generated candidate does not exist", code="GENERATION_CASE_NOT_FOUND"
                )
            if case.risk in {"high", "critical"}:
                raise GenerationWorkspaceError(
                    "high and critical risk candidates require individual approval",
                    code="GENERATION_HIGH_RISK_BATCH_BLOCKED",
                )
        decisions: list[ApprovalDecision] = []
        for case_id in case_ids:
            case = candidates[case_id]
            decisions.append(
                await self.review(
                    session_id=session_id,
                    case_id=case_id,
                    subject_hash=case.current_spec_hash,
                    decision="APPROVED",
                    reviewer=reviewer,
                    comment=comment,
                )
            )
        return decisions


__all__ = ["GenerationWorkbenchService", "GenerationWorkspaceError"]
