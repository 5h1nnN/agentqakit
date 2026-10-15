"""Campaign builder, confirmation, execution, and persisted progress."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import tempfile
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import yaml
from pydantic import JsonValue

from agentqa.api.models import CampaignCreateRequest
from agentqa.application import RunExecutionConfig, execute_run
from agentqa.conventions import new_id, utc_now
from agentqa.errors import AgentQAError
from agentqa.evaluators import JudgeProviderLike
from agentqa.manifest import capture_run_manifest
from agentqa.models import AgentManifest, AgentTestSpec, ErrorInfo, MutationSpec, TargetSpec
from agentqa.models.specs import TargetSpecBase
from agentqa.specs import hash_spec, load_specs, normalize_spec
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.storage import ResultStore
from agentqa.workbench.campaign_store import CampaignStore
from agentqa.workbench.catalog import ArtifactCatalogService
from agentqa.workbench.generation import GenerationWorkbenchService
from agentqa.workbench.models import (
    ApprovalDecision,
    ArtifactRef,
    Campaign,
    CampaignConfig,
    CampaignConfirmationChallenge,
    CampaignRiskSummary,
    CampaignSubrun,
    CampaignTarget,
    CampaignTestSelection,
    WorkbenchProject,
    WorkflowStep,
    WorkflowStepKind,
)
from agentqa.workbench.orchestrator import WorkflowOrchestrator
from agentqa.workbench.store import WorkflowStore

CAMPAIGN_ARTIFACT_SCHEMA_VERSION = "agentqa.campaign/v1alpha1"


class CampaignError(AgentQAError):
    def __init__(self, message: str, *, code: str = "CAMPAIGN_ERROR") -> None:
        super().__init__(message, code=code)


def _safe_project_path(workspace_root: Path, project: WorkbenchProject, raw: str) -> Path:
    project_root = (workspace_root / PurePosixPath(project.workspace_path)).resolve()
    candidate = (project_root / PurePosixPath(raw)).resolve()
    try:
        project_root.relative_to(workspace_root)
        candidate.relative_to(project_root)
    except ValueError:
        raise CampaignError(
            "campaign path escapes the selected project", code="CAMPAIGN_PATH_ERROR"
        ) from None
    if not candidate.exists():
        raise CampaignError("campaign path does not exist", code="CAMPAIGN_PATH_NOT_FOUND")
    return candidate


def _safe_replay_path(workspace_root: Path, project: WorkbenchProject, raw: str) -> Path:
    """Allow a project file or an existing workspace-level run cassette."""
    try:
        return _safe_project_path(workspace_root, project, raw)
    except CampaignError as project_error:
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or "\\" in raw:
            raise project_error from None
        candidate = (workspace_root / relative).resolve()
        try:
            candidate.relative_to(workspace_root)
        except ValueError:
            raise project_error from None
        if not candidate.exists():
            raise project_error from None
        return candidate


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _write_immutable(path: Path, payload: bytes) -> str:
    digest = sha256_digest(payload)
    checksum = path.with_name(f"{path.name}.sha256")
    if path.exists() or checksum.exists():
        existing = path.read_bytes() if path.exists() else b""
        if existing != payload:
            raise CampaignError(
                "immutable campaign input already exists with different content",
                code="CAMPAIGN_INPUT_CONFLICT",
            )
        return digest
    _atomic_write(path, payload)
    _atomic_write(checksum, f"{digest}\n".encode())
    return digest


def _only(items: list[Any], label: str) -> Any:
    if len(items) != 1:
        raise CampaignError(
            f"{label} must contain exactly one matching specification",
            code="CAMPAIGN_SPEC_SELECTION_ERROR",
        )
    return items[0]


class CampaignWorkbenchService:
    """Turn reviewed inputs into safe, separately persisted target runs."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        artifact_directory: Path,
        result_store: ResultStore,
        workflow_store: WorkflowStore,
        campaign_store: CampaignStore,
        generation: GenerationWorkbenchService,
        artifact_catalog: ArtifactCatalogService,
        orchestrator: WorkflowOrchestrator,
        judge_provider: JudgeProviderLike | None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.artifact_directory = artifact_directory.resolve()
        self.result_store = result_store
        self.workflow_store = workflow_store
        self.campaign_store = campaign_store
        self.generation = generation
        self.artifact_catalog = artifact_catalog
        self.orchestrator = orchestrator
        self.judge_provider = judge_provider

    async def _context(self, workflow_id: str) -> tuple[Any, WorkbenchProject]:
        workflow = await self.workflow_store.get_workflow(workflow_id)
        if workflow is None:
            raise CampaignError("workflow does not exist", code="WORKFLOW_NOT_FOUND")
        project = await self.workflow_store.get_project(workflow.project_id)
        if project is None:
            raise CampaignError("project does not exist", code="WORKBENCH_PROJECT_NOT_FOUND")
        return workflow, project

    async def inputs(self, workflow_id: str) -> dict[str, JsonValue]:
        """Return bounded selectable inputs for the Web builder."""
        _, project = await self._context(workflow_id)
        golden_ref = project.default_spec_refs.get("tests")
        golden: list[dict[str, JsonValue]] = []
        if golden_ref is not None:
            path = _safe_project_path(self.workspace_root, project, golden_ref)
            golden = [
                {
                    "caseId": item.spec.metadata.name,
                    "risk": item.spec.metadata.labels.get("risk", "low"),
                    "specHash": hash_spec(item.spec),
                    "sourceRef": item.source.relative_to(
                        (self.workspace_root / project.workspace_path).resolve()
                    ).as_posix(),
                }
                for item in load_specs(path)
                if isinstance(item.spec, AgentTestSpec)
            ]
        generated: list[dict[str, JsonValue]] = []
        for session in await self.generation.session_store.list_sessions(workflow_id):
            if session.status != "CANDIDATES_READY":
                continue
            for candidate in await self.generation.candidates(session.session_id):
                generated.append(
                    {
                        "sessionId": session.session_id,
                        "caseId": candidate.case_id,
                        "risk": candidate.risk,
                        "specHash": candidate.current_spec_hash,
                        "campaignEligible": candidate.campaign_eligible,
                        "reviewStatus": candidate.review_status,
                    }
                )
        targets = [
            {"role": key.removesuffix("_target"), "path": value}
            for key, value in project.default_spec_refs.items()
            if key == "target" or key.endswith("_target")
        ]
        return cast(
            dict[str, JsonValue],
            {
                "goldenTestsPath": golden_ref,
                "golden": golden,
                "generated": generated,
                "targets": targets,
                "manifestPath": project.default_spec_refs.get("manifest"),
                "mutationsPath": project.default_spec_refs.get("mutations"),
                "judge": {
                    "available": self.judge_provider is not None,
                    "provider": None if self.judge_provider is None else self.judge_provider.name,
                    "model": None if self.judge_provider is None else self.judge_provider.model,
                },
            },
        )

    async def create(self, workflow_id: str, request: CampaignCreateRequest) -> Campaign:
        workflow, project = await self._context(workflow_id)
        if workflow.status not in {"READY", "INTERRUPTED"}:
            raise CampaignError(
                "workflow must be READY before creating a campaign",
                code="CAMPAIGN_WORKFLOW_STATE_ERROR",
            )
        manifest_ref = request.manifest_path or project.default_spec_refs.get("manifest")
        if manifest_ref is None:
            raise CampaignError("campaign requires a manifest", code="CAMPAIGN_MANIFEST_REQUIRED")
        if project.last_manifest_snapshot is None:
            raise CampaignError(
                "run project preflight before creating a campaign",
                code="CAMPAIGN_PREFLIGHT_REQUIRED",
            )
        manifest_path = _safe_project_path(self.workspace_root, project, manifest_ref)
        manifest = cast(
            AgentManifest,
            _only(
                [
                    item.spec
                    for item in load_specs(manifest_path)
                    if isinstance(item.spec, AgentManifest)
                ],
                "manifest path",
            ),
        )
        manifest_hash = hash_spec(manifest)
        if project.last_manifest_snapshot.declared_manifest_hash != manifest_hash:
            raise CampaignError(
                "manifest changed after preflight; run preflight again",
                code="CAMPAIGN_MANIFEST_STALE",
            )
        current_manifest_snapshot = capture_run_manifest(
            manifest,
            workspace=(self.workspace_root / project.workspace_path).resolve(),
        )
        await self.workflow_store.update_project_observation(
            project.project_id, current_manifest_snapshot
        )

        selected_specs: list[tuple[CampaignTestSelection, AgentTestSpec]] = []
        golden_ref = request.golden_tests_path or project.default_spec_refs.get("tests")
        if golden_ref is not None and (request.golden_case_ids or not request.generated_tests):
            golden_path = _safe_project_path(self.workspace_root, project, golden_ref)
            available = {
                item.spec.metadata.name: (item.source, item.spec)
                for item in load_specs(golden_path)
                if isinstance(item.spec, AgentTestSpec)
            }
            chosen = request.golden_case_ids or list(available)
            missing = sorted(set(chosen) - set(available))
            if missing:
                raise CampaignError(
                    "unknown golden test(s): " + ", ".join(missing),
                    code="CAMPAIGN_TEST_NOT_FOUND",
                )
            for case_id in chosen:
                source, spec = available[case_id]
                selected_specs.append(
                    (
                        CampaignTestSelection(
                            source="golden",
                            case_id=case_id,
                            spec_hash=hash_spec(spec),
                            source_ref=source.relative_to(
                                (self.workspace_root / project.workspace_path).resolve()
                            ).as_posix(),
                            risk=cast(
                                Any,
                                spec.metadata.labels.get("risk", "low").lower()
                                if spec.metadata.labels.get("risk", "low").lower()
                                in {"low", "medium", "high", "critical"}
                                else "low",
                            ),
                        ),
                        spec,
                    )
                )
        for selection in request.generated_tests:
            session = await self.generation.require_session(selection.session_id)
            if session.workflow_id != workflow_id:
                raise CampaignError(
                    "generated test belongs to another workflow",
                    code="CAMPAIGN_GENERATED_TEST_SCOPE_ERROR",
                )
            candidate = next(
                (
                    item
                    for item in await self.generation.candidates(selection.session_id)
                    if item.case_id == selection.case_id
                ),
                None,
            )
            if (
                candidate is None
                or candidate.current_spec_hash != selection.spec_hash
                or not candidate.campaign_eligible
            ):
                raise CampaignError(
                    "generated test revision is missing, stale, or not approved",
                    code="CAMPAIGN_GENERATED_TEST_NOT_ELIGIBLE",
                )
            selected_specs.append(
                (
                    CampaignTestSelection(
                        source="generated",
                        case_id=candidate.case_id,
                        spec_hash=candidate.current_spec_hash,
                        source_ref=f"generation:{selection.session_id}:{candidate.case_id}",
                        risk=candidate.risk,
                    ),
                    candidate.spec,
                )
            )
        if not selected_specs:
            raise CampaignError("select at least one test", code="CAMPAIGN_TESTS_REQUIRED")
        case_ids = [item.case_id for item, _ in selected_specs]
        if len(case_ids) != len(set(case_ids)):
            raise CampaignError(
                "selected tests must have unique case IDs", code="CAMPAIGN_DUPLICATE_TEST"
            )

        targets: list[CampaignTarget] = []
        for target_request in request.targets:
            path = _safe_project_path(self.workspace_root, project, target_request.path)
            target = cast(
                TargetSpec,
                _only(
                    [
                        item.spec
                        for item in load_specs(path)
                        if isinstance(item.spec, TargetSpecBase)
                    ],
                    "target path",
                ),
            )
            targets.append(
                CampaignTarget(
                    role=target_request.role,
                    path=target_request.path,
                    name=target.metadata.name,
                    spec_hash=hash_spec(target),
                )
            )
        if len({(item.role, item.path) for item in targets}) != len(targets):
            raise CampaignError("campaign targets must be unique", code="CAMPAIGN_DUPLICATE_TARGET")

        mutation_ref = request.mutations_path
        mutation_ids: list[str] = []
        if mutation_ref is not None:
            mutation_path = _safe_project_path(self.workspace_root, project, mutation_ref)
            mutation_ids = [
                item.spec.metadata.name
                for item in load_specs(mutation_path)
                if isinstance(item.spec, MutationSpec)
            ]
            if not mutation_ids:
                raise CampaignError(
                    "mutation path contains no Mutation documents",
                    code="CAMPAIGN_MUTATIONS_REQUIRED",
                )
        cassette_ref = request.replay_cassette
        if request.replay_mode in {"replay", "hybrid"}:
            if cassette_ref is None:
                raise CampaignError(
                    "replay and hybrid modes require a cassette",
                    code="CAMPAIGN_CASSETTE_REQUIRED",
                )
            cassette = _safe_replay_path(self.workspace_root, project, cassette_ref)
            if not cassette.is_file():
                raise CampaignError(
                    "replay cassette must be an existing file",
                    code="CAMPAIGN_CASSETTE_REQUIRED",
                )
        if request.judge_enabled and self.judge_provider is None:
            raise CampaignError(
                "DeepSeek Judge is not configured on this server",
                code="CAMPAIGN_JUDGE_UNAVAILABLE",
            )

        side_effect_tools = [tool.name for tool in manifest.tools if tool.side_effect != "none"]
        external = [tool.name for tool in manifest.tools if tool.side_effect == "external-write"]
        irreversible = [tool.name for tool in manifest.tools if tool.side_effect == "irreversible"]
        high_risk = [
            selection.case_id
            for selection, _ in selected_specs
            if selection.risk in {"high", "critical"}
        ]
        reasons: list[str] = []
        if high_risk:
            reasons.append("high or critical risk tests")
        if mutation_ids:
            reasons.append("mutation campaign")
        if external or irreversible:
            reasons.append("external or irreversible side effects")
            if not request.allow_side_effects:
                raise CampaignError(
                    "external side effects require allowSideEffects=true",
                    code="CAMPAIGN_SIDE_EFFECTS_NOT_ALLOWED",
                )
        config = CampaignConfig(
            tests=[item for item, _ in selected_specs],
            targets=targets,
            manifest_path=manifest_ref,
            manifest_hash=manifest_hash,
            manifest_snapshot=current_manifest_snapshot,
            mutations_path=mutation_ref,
            mutation_ids=mutation_ids,
            replay_mode=request.replay_mode,
            replay_cassette=cassette_ref,
            judge_enabled=request.judge_enabled,
            judge_provider=(
                None if not request.judge_enabled else cast(Any, self.judge_provider).name
            ),
            trials=request.trials,
            random_seed=request.random_seed,
            concurrency=request.concurrency,
            item_timeout_seconds=request.item_timeout_seconds,
            evaluator_timeout_seconds=request.evaluator_timeout_seconds,
            max_total_tokens=request.max_total_tokens,
            max_total_cost_usd=request.max_total_cost_usd,
            max_duration_seconds=request.max_duration_seconds,
            allow_side_effects=request.allow_side_effects,
        )
        config_hash = sha256_digest(canonical_json_bytes(config))
        multiplier = max(1, len(mutation_ids))
        per_target_trials = sum(
            (request.trials or spec.trials) * multiplier for _, spec in selected_specs
        )
        risk = CampaignRiskSummary(
            case_count=len(selected_specs),
            target_count=len(targets),
            mutation_count=len(mutation_ids),
            max_trials=per_target_trials * len(targets),
            high_risk_cases=high_risk,
            side_effect_tools=side_effect_tools,
            external_side_effects=external,
            irreversible_side_effects=irreversible,
            requires_confirmation=bool(reasons),
            reasons=reasons,
        )
        campaign_id = new_id()
        now = utc_now()
        campaign = Campaign(
            campaign_id=campaign_id,
            workflow_id=workflow_id,
            project_id=project.project_id,
            idempotency_key=request.idempotency_key,
            config_hash=config_hash,
            config=config,
            risk=risk,
            status="AWAITING_CONFIRMATION" if risk.requires_confirmation else "DRAFT",
            total_trials=risk.max_trials,
            created_at=now,
            updated_at=now,
            subruns=[
                CampaignSubrun(
                    subrun_id=new_id(),
                    campaign_id=campaign_id,
                    target=target,
                    run_id=new_id(),
                    status="QUEUED",
                    planned_trials=per_target_trials,
                )
                for target in targets
            ],
        )
        stored = await self.campaign_store.create(campaign)
        await self._freeze_inputs(stored, selected_specs)
        if stored.campaign_id == campaign_id:
            await self.workflow_store.record_event(
                workflow_id,
                event_type="CAMPAIGN_CREATED",
                message="Immutable campaign configuration created.",
                data={
                    "campaign_id": campaign_id,
                    "config_hash": config_hash,
                    "requires_confirmation": risk.requires_confirmation,
                    "max_trials": risk.max_trials,
                },
            )
        return stored

    async def _freeze_inputs(
        self, campaign: Campaign, selected_specs: list[tuple[CampaignTestSelection, AgentTestSpec]]
    ) -> None:
        root = self.artifact_directory / "campaigns" / campaign.campaign_id / "inputs"
        for selection, spec in selected_specs:
            payload = yaml.safe_dump(
                normalize_spec(spec), allow_unicode=True, sort_keys=False
            ).encode()
            _write_immutable(root / "selected" / f"{selection.case_id}.yaml", payload)
        config_payload = (
            canonical_json_bytes(
                {
                    "schema_version": CAMPAIGN_ARTIFACT_SCHEMA_VERSION,
                    "campaign_id": campaign.campaign_id,
                    "config_hash": campaign.config_hash,
                    "config": campaign.config.model_dump(mode="json"),
                    "risk": campaign.risk.model_dump(mode="json"),
                }
            )
            + b"\n"
        )
        _write_immutable(root / "campaign-config.json", config_payload)

    async def challenge(self, campaign_id: str) -> dict[str, Any]:
        campaign = await self.require(campaign_id)
        if not campaign.risk.requires_confirmation:
            raise CampaignError(
                "campaign does not require confirmation", code="CAMPAIGN_CONFIRMATION_NOT_REQUIRED"
            )
        if campaign.status != "AWAITING_CONFIRMATION":
            raise CampaignError(
                "campaign is not awaiting confirmation", code="CAMPAIGN_CONFIRMATION_STATE_ERROR"
            )
        phrase = f"CONFIRM-{secrets.token_hex(4).upper()}"
        now = utc_now()
        challenge = CampaignConfirmationChallenge(
            challenge_id=new_id(),
            campaign_id=campaign.campaign_id,
            config_hash=campaign.config_hash,
            phrase_hash=sha256_digest(phrase.encode()),
            status="PENDING",
            created_at=now,
            expires_at=now + timedelta(minutes=10),
        )
        await self.campaign_store.save_challenge(challenge)
        return {"challenge": challenge, "phrase": phrase, "risk": campaign.risk}

    async def confirm(
        self,
        campaign_id: str,
        *,
        challenge_id: str,
        phrase: str,
        config_hash: str,
        reviewer: str,
        comment: str,
    ) -> Campaign:
        campaign = await self.require(campaign_id)
        if config_hash != campaign.config_hash:
            raise CampaignError(
                "confirmation config hash is stale", code="CAMPAIGN_CONFIG_HASH_MISMATCH"
            )
        await self.campaign_store.consume_challenge(
            challenge_id,
            campaign_id,
            config_hash,
            sha256_digest(phrase.encode()),
        )
        decision = ApprovalDecision(
            approval_id=new_id(),
            workflow_id=campaign.workflow_id,
            subject_kind="CampaignConfig",
            subject_id=campaign.campaign_id,
            subject_hash=campaign.config_hash,
            decision="APPROVED",
            reviewer=reviewer,
            comment=comment,
            created_at=utc_now(),
        )
        await self.workflow_store.save_approval(decision)
        updated = campaign.model_copy(
            update={"status": "DRAFT", "approval_id": decision.approval_id, "updated_at": utc_now()}
        )
        await self.campaign_store.update(updated)
        await self.workflow_store.record_event(
            campaign.workflow_id,
            event_type="CAMPAIGN_APPROVED",
            message="One-time confirmation accepted for the immutable campaign config.",
            data={"campaign_id": campaign_id, "approval_id": decision.approval_id},
        )
        return updated

    async def start(
        self, campaign_id: str, *, config_hash: str, idempotency_key: str
    ) -> tuple[Campaign, WorkflowStep]:
        campaign = await self.require(campaign_id)
        if campaign.config_hash != config_hash:
            raise CampaignError(
                "campaign config hash changed; confirmation is invalid",
                code="CAMPAIGN_CONFIG_HASH_MISMATCH",
            )
        if campaign.status not in {"DRAFT", "INTERRUPTED", "INCOMPLETE"}:
            raise CampaignError(
                "campaign cannot be started from its current state",
                code="CAMPAIGN_STATE_ERROR",
            )
        if campaign.risk.requires_confirmation:
            approvals = await self.workflow_store.list_approvals(campaign.workflow_id)
            valid = any(
                item.approval_id == campaign.approval_id
                and item.subject_kind == "CampaignConfig"
                and item.subject_id == campaign.campaign_id
                and item.subject_hash == campaign.config_hash
                and item.decision == "APPROVED"
                for item in approvals
            )
            if not valid:
                raise CampaignError(
                    "campaign requires a valid approval bound to this config hash",
                    code="CAMPAIGN_APPROVAL_REQUIRED",
                )
        queued = campaign.model_copy(
            update={"status": "QUEUED", "updated_at": utc_now(), "finished_at": None, "error": None}
        )
        await self.campaign_store.update(queued)
        kind: WorkflowStepKind = (
            "RUN_MUTATIONS"
            if campaign.config.mutation_ids
            else ("RECORD_REPLAY" if campaign.config.replay_mode != "off" else "RUN_CAMPAIGN")
        )
        step = await self.orchestrator.submit(
            workflow_id=campaign.workflow_id,
            kind=kind,
            idempotency_key=idempotency_key,
            input_data={"campaign_id": campaign_id, "config_hash": campaign.config_hash},
            handler=lambda workflow_step: self._execute(queued, workflow_step),
        )
        return await self.require(campaign.campaign_id), step

    async def _execute(self, initial: Campaign, step: WorkflowStep) -> list[ArtifactRef]:
        campaign = await self.require(initial.campaign_id)
        running = campaign.model_copy(
            update={
                "status": "RUNNING",
                "workflow_step_id": step.step_id,
                "started_at": utc_now(),
                "updated_at": utc_now(),
            }
        )
        await self.campaign_store.update(running)
        artifacts: list[ArtifactRef] = []
        started = asyncio.get_running_loop().time()
        try:
            for stored_subrun in running.subruns:
                if stored_subrun.status == "COMPLETED":
                    continue
                if running.config.max_duration_seconds is not None:
                    remaining = running.config.max_duration_seconds - (
                        asyncio.get_running_loop().time() - started
                    )
                    if remaining <= 0:
                        await self._mark_incomplete(
                            running.campaign_id, "campaign time budget exhausted"
                        )
                        break
                else:
                    remaining = None
                subrun = stored_subrun
                if subrun.status in {"INTERRUPTED", "FAILED", "INCOMPLETE", "CANCELLED"}:
                    subrun = subrun.model_copy(
                        update={
                            "prior_run_ids": [*subrun.prior_run_ids, subrun.run_id],
                            "run_id": new_id(),
                            "attempt": subrun.attempt + 1,
                            "status": "QUEUED",
                            "completed_trials": 0,
                            "current_case_id": None,
                            "replay_hits": 0,
                            "replay_misses": 0,
                            "live_fallbacks": 0,
                            "judge_evaluations": 0,
                            "mutation_applications": 0,
                            "mutation_killed": 0,
                            "mutation_survived": 0,
                            "mutation_not_reached": 0,
                            "mutation_invalid": 0,
                            "summary": None,
                            "budget": None,
                            "artifact_path": None,
                            "started_at": None,
                            "finished_at": None,
                            "error": None,
                        }
                    )
                    await self.campaign_store.update_subrun(subrun)
                try:
                    artifact = await self._execute_subrun(running, subrun, remaining)
                    artifacts.append(artifact)
                except TimeoutError:
                    now = utc_now()
                    current = await self.require(running.campaign_id)
                    timed_out_subrun = next(
                        item for item in current.subruns if item.subrun_id == subrun.subrun_id
                    )
                    await self.campaign_store.update_subrun(
                        timed_out_subrun.model_copy(
                            update={
                                "status": "INCOMPLETE",
                                "finished_at": now,
                                "error": ErrorInfo(
                                    code="CAMPAIGN_TIME_BUDGET_EXCEEDED",
                                    message="campaign time budget expired during this target run",
                                ),
                            }
                        )
                    )
                    await self._mark_incomplete(
                        running.campaign_id, "campaign time budget expired during target execution"
                    )
                    break
                except Exception as exc:
                    await self._read_live_progress(running.campaign_id, subrun)
                    current = await self.require(running.campaign_id)
                    failed_subrun = next(
                        item for item in current.subruns if item.subrun_id == subrun.subrun_id
                    )
                    await self.campaign_store.update_subrun(
                        failed_subrun.model_copy(
                            update={
                                "status": "FAILED",
                                "finished_at": utc_now(),
                                "error": ErrorInfo(
                                    code=getattr(exc, "code", "CAMPAIGN_SUBRUN_ERROR"),
                                    message=getattr(
                                        exc,
                                        "message",
                                        f"target run raised {type(exc).__name__}",
                                    ),
                                ),
                            }
                        )
                    )
                    raise
            refreshed = await self.require(running.campaign_id)
            if refreshed.status == "RUNNING":
                terminal = (
                    "COMPLETED"
                    if all(item.status == "COMPLETED" for item in refreshed.subruns)
                    else "FAILED"
                )
                refreshed = refreshed.model_copy(
                    update={
                        "status": terminal,
                        "finished_at": utc_now(),
                        "updated_at": utc_now(),
                        "completed_trials": sum(
                            item.completed_trials for item in refreshed.subruns
                        ),
                    }
                )
                await self.campaign_store.update(refreshed)
            summary_ref = await self._write_summary(await self.require(running.campaign_id))
            artifacts.append(summary_ref)
            await self.workflow_store.record_event(
                running.workflow_id,
                event_type="CAMPAIGN_FINISHED",
                message="Campaign reached a persisted terminal state.",
                data={
                    "campaign_id": running.campaign_id,
                    "status": (await self.require(running.campaign_id)).status,
                },
            )
            return artifacts
        except asyncio.CancelledError:
            now = utc_now()
            refreshed = await self.require(running.campaign_id)
            for subrun in refreshed.subruns:
                if subrun.status in {"QUEUED", "RUNNING"}:
                    await self.campaign_store.update_subrun(
                        subrun.model_copy(update={"status": "INTERRUPTED", "finished_at": now})
                    )
            await self.campaign_store.update(
                refreshed.model_copy(
                    update={"status": "INTERRUPTED", "finished_at": now, "updated_at": now}
                )
            )
            raise
        except Exception as exc:
            refreshed = await self.require(running.campaign_id)
            now = utc_now()
            await self.campaign_store.update(
                refreshed.model_copy(
                    update={
                        "status": "FAILED",
                        "finished_at": now,
                        "updated_at": now,
                        "error": ErrorInfo(
                            code=getattr(exc, "code", "CAMPAIGN_EXECUTION_ERROR"),
                            message=getattr(
                                exc, "message", f"campaign raised {type(exc).__name__}"
                            ),
                        ),
                    }
                )
            )
            raise

    async def _execute_subrun(
        self, campaign: Campaign, subrun: CampaignSubrun, timeout: float | None
    ) -> ArtifactRef:
        now = utc_now()
        running = subrun.model_copy(update={"status": "RUNNING", "started_at": now})
        await self.campaign_store.update_subrun(running)
        await self._refresh_campaign_progress(campaign.campaign_id)
        tests_path = await self._materialize_target_suite(campaign, running.target)
        _, project = await self._context(campaign.workflow_id)
        target_path = _safe_project_path(self.workspace_root, project, running.target.path)
        manifest_path = _safe_project_path(
            self.workspace_root, project, campaign.config.manifest_path
        )
        mutation_path = (
            None
            if campaign.config.mutations_path is None
            else _safe_project_path(self.workspace_root, project, campaign.config.mutations_path)
        )
        cassette_path = (
            None
            if campaign.config.replay_cassette is None
            else _safe_replay_path(self.workspace_root, project, campaign.config.replay_cassette)
        )
        replay_mode: Literal["off", "record", "replay"]
        if campaign.config.replay_mode in {"replay", "hybrid"}:
            replay_mode = "replay"
        elif campaign.config.replay_mode == "record":
            replay_mode = "record"
        else:
            replay_mode = "off"
        execute_task = asyncio.create_task(
            execute_run(
                RunExecutionConfig(
                    tests_path=tests_path,
                    target_path=target_path,
                    mutations_path=mutation_path,
                    manifest_path=manifest_path,
                    artifact_directory=self.artifact_directory,
                    random_seed=campaign.config.random_seed,
                    concurrency=campaign.config.concurrency,
                    max_total_tokens=campaign.config.max_total_tokens,
                    max_total_cost_usd=campaign.config.max_total_cost_usd,
                    item_timeout_seconds=campaign.config.item_timeout_seconds,
                    evaluator_timeout_seconds=campaign.config.evaluator_timeout_seconds,
                    trace_receiver=False,
                    replay_mode=replay_mode,
                    replay_cassette=cassette_path,
                    allow_live_fallback=campaign.config.replay_mode == "hybrid",
                    generate_failure_bundles=True,
                    original_command=f"AgentQAKit Web campaign {campaign.campaign_id}",
                    run_id=running.run_id,
                    judge_provider=(self.judge_provider if campaign.config.judge_enabled else None),
                )
            ),
            name=f"agentqa-campaign-run-{running.run_id}",
        )
        try:
            deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
            while not execute_task.done():
                wait_for = 0.5
                if deadline is not None:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        execute_task.cancel()
                        await asyncio.gather(execute_task, return_exceptions=True)
                        raise TimeoutError
                    wait_for = min(wait_for, remaining)
                done, _ = await asyncio.wait({execute_task}, timeout=wait_for)
                await self._read_live_progress(campaign.campaign_id, running)
                if done:
                    break
            result = await execute_task
        finally:
            await self._read_live_progress(campaign.campaign_id, running)
        artifact_path = self.artifact_directory / "runs" / result.run_id
        await self.result_store.save_run(
            result, project_id=campaign.project_id, artifact_path=artifact_path
        )
        registered = await self.artifact_catalog.register_directory(
            campaign.project_id, artifact_path, source="WORKFLOW"
        )
        replay_hits = sum(item.replay.hits for item in result.results if item.replay is not None)
        replay_misses = sum(
            item.replay.misses for item in result.results if item.replay is not None
        )
        live_fallbacks = sum(
            item.replay.live_fallbacks for item in result.results if item.replay is not None
        )
        judge_evaluations = sum(
            evaluation.evaluator_type == "llm_judge"
            for item in result.results
            for evaluation in item.evaluations
        )
        mutation_applications = sum(len(item.mutation_applications) for item in result.results)
        score = result.mutation_score
        finished = running.model_copy(
            update={
                "status": result.status,
                "completed_trials": result.summary.completed_trials,
                "current_case_id": None,
                "replay_hits": replay_hits,
                "replay_misses": replay_misses,
                "live_fallbacks": live_fallbacks,
                "judge_evaluations": judge_evaluations,
                "mutation_applications": mutation_applications,
                "mutation_killed": 0 if score is None else score.overall.killed,
                "mutation_survived": 0 if score is None else score.overall.survived,
                "mutation_not_reached": 0 if score is None else score.overall.not_reached,
                "mutation_invalid": 0 if score is None else score.overall.invalid,
                "summary": result.summary,
                "budget": result.budget,
                "artifact_path": str(artifact_path),
                "finished_at": utc_now(),
                "error": result.error,
            }
        )
        await self.campaign_store.update_subrun(finished)
        await self._refresh_campaign_progress(campaign.campaign_id)
        summary_artifact = next(
            (item for item in registered if Path(item.path).name == "summary.json"), None
        )
        if summary_artifact is None:
            raise CampaignError("run summary was not registered", code="CAMPAIGN_ARTIFACT_MISSING")
        return ArtifactRef(
            artifact_id=summary_artifact.artifact_id,
            artifact_type="RunSummary",
            uri=summary_artifact.path,
            schema_version=summary_artifact.schema_version,
            checksum=summary_artifact.checksum,
            source="workflow",
        )

    async def _read_live_progress(self, campaign_id: str, subrun: CampaignSubrun) -> None:
        path = self.artifact_directory / "runs" / subrun.run_id / "results.jsonl"
        completed = 0
        current: str | None = None
        replay_hits = replay_misses = live_fallbacks = 0
        judge_evaluations = mutation_applications = 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                data = record.get("data", {})
                completed += 1
                current = data.get("test_case_id")
                replay = data.get("replay") or {}
                replay_hits += int(replay.get("hits", 0))
                replay_misses += int(replay.get("misses", 0))
                live_fallbacks += int(replay.get("live_fallbacks", 0))
                judge_evaluations += sum(
                    evaluation.get("evaluator_type") == "llm_judge"
                    for evaluation in data.get("evaluations", [])
                    if isinstance(evaluation, dict)
                )
                applications = data.get("mutation_applications", [])
                if isinstance(applications, list):
                    mutation_applications += len(applications)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        try:
            resolved = json.loads((path.parent / "resolved-tests.json").read_text(encoding="utf-8"))
            sequence = [
                str(spec["metadata"]["name"])
                for spec in resolved.get("tests", [])
                for _ in range(int(spec.get("trials", 1)))
            ]
            current = sequence[completed] if completed < len(sequence) else None
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass
        fresh = await self.require(campaign_id)
        selected = next(
            (item for item in fresh.subruns if item.subrun_id == subrun.subrun_id), None
        )
        if selected is None or selected.status != "RUNNING":
            return
        await self.campaign_store.update_subrun(
            selected.model_copy(
                update={
                    "completed_trials": min(completed, selected.planned_trials),
                    "current_case_id": current,
                    "replay_hits": replay_hits,
                    "replay_misses": replay_misses,
                    "live_fallbacks": live_fallbacks,
                    "judge_evaluations": judge_evaluations,
                    "mutation_applications": mutation_applications,
                }
            )
        )
        await self._refresh_campaign_progress(campaign_id)

    async def _refresh_campaign_progress(self, campaign_id: str) -> Campaign:
        campaign = await self.require(campaign_id)
        updated = campaign.model_copy(
            update={
                "completed_trials": sum(item.completed_trials for item in campaign.subruns),
                "updated_at": utc_now(),
            }
        )
        return await self.campaign_store.update(updated)

    async def _mark_incomplete(self, campaign_id: str, message: str) -> None:
        campaign = await self.require(campaign_id)
        now = utc_now()
        for subrun in campaign.subruns:
            if subrun.status == "QUEUED":
                await self.campaign_store.update_subrun(
                    subrun.model_copy(update={"status": "INCOMPLETE", "finished_at": now})
                )
        await self.campaign_store.update(
            campaign.model_copy(
                update={
                    "status": "INCOMPLETE",
                    "finished_at": now,
                    "updated_at": now,
                    "error": ErrorInfo(code="CAMPAIGN_INCOMPLETE", message=message),
                }
            )
        )

    async def _materialize_target_suite(self, campaign: Campaign, target: CampaignTarget) -> Path:
        selected_dir = (
            self.artifact_directory / "campaigns" / campaign.campaign_id / "inputs" / "selected"
        )
        specs = [
            item.spec for item in load_specs(selected_dir) if isinstance(item.spec, AgentTestSpec)
        ]
        materialized: list[AgentTestSpec] = []
        mutation_ids: list[str | None] = (
            list(campaign.config.mutation_ids) if campaign.config.mutation_ids else [None]
        )
        for spec in specs:
            for mutation_id in mutation_ids:
                name = spec.metadata.name
                if mutation_id is not None:
                    raw_name = f"{name}--{mutation_id}"
                    name = (
                        raw_name
                        if len(raw_name) <= 128
                        else f"{raw_name[:115]}-{sha256_digest(raw_name.encode())[-12:]}"
                    )
                materialized.append(
                    spec.model_copy(
                        update={
                            "metadata": spec.metadata.model_copy(update={"name": name}),
                            "target_ref": target.name,
                            "mutation": mutation_id if mutation_id is not None else spec.mutation,
                            "trials": campaign.config.trials or spec.trials,
                        }
                    )
                )
        output = (
            self.artifact_directory
            / "campaigns"
            / campaign.campaign_id
            / "inputs"
            / "targets"
            / target.role
            / target.name
        )
        for spec in materialized:
            payload = yaml.safe_dump(
                normalize_spec(spec), allow_unicode=True, sort_keys=False
            ).encode()
            _write_immutable(output / f"{spec.metadata.name}.yaml", payload)
        return output

    async def _write_summary(self, campaign: Campaign) -> ArtifactRef:
        identity = campaign.workflow_step_id or campaign.campaign_id
        path = (
            self.artifact_directory
            / "campaigns"
            / campaign.campaign_id
            / f"campaign-{identity}.json"
        )
        payload = (
            canonical_json_bytes(
                {
                    "schema_version": CAMPAIGN_ARTIFACT_SCHEMA_VERSION,
                    **campaign.model_dump(mode="json"),
                }
            )
            + b"\n"
        )
        checksum = _write_immutable(path, payload)
        artifact = await self.artifact_catalog.register_path(
            campaign.project_id, path, source="WORKFLOW", workflow_id=campaign.workflow_id
        )
        return ArtifactRef(
            artifact_id=artifact.artifact_id,
            artifact_type="Campaign",
            uri=artifact.path,
            schema_version=CAMPAIGN_ARTIFACT_SCHEMA_VERSION,
            checksum=checksum,
            source="workflow",
        )

    async def require(self, campaign_id: str) -> Campaign:
        campaign = await self.campaign_store.get(campaign_id)
        if campaign is None:
            raise CampaignError("campaign does not exist", code="CAMPAIGN_NOT_FOUND")
        return campaign


__all__ = ["CAMPAIGN_ARTIFACT_SCHEMA_VERSION", "CampaignError", "CampaignWorkbenchService"]
