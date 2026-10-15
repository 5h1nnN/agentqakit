"""T21 failure analysis, external repair, comparison, gate, and report workflow."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import JsonValue, ValidationError

from agentqa.analysis.evidence import build_evidence_bundle, evidence_bundle_hash
from agentqa.analysis.io import load_failure_analysis_input, write_analysis_report
from agentqa.application import FailureAnalysisService
from agentqa.comparison import compare_runs, load_run_artifact
from agentqa.comparison.io import write_report
from agentqa.conventions import new_id, utc_now
from agentqa.errors import AgentQAError
from agentqa.gates import evaluate_gate
from agentqa.llm import StructuredLLMProvider
from agentqa.models import (
    AnalysisReport,
    ComparisonReport,
    GateResult,
    GenerationReport,
    QualityGateSpec,
)
from agentqa.models.generation import CodeEvidence
from agentqa.specs import SpecLoadError, load_spec_file
from agentqa.specs.hashing import canonical_json_bytes, hash_spec, sha256_digest
from agentqa.storage import ResultStore
from agentqa.storage.redaction import redact_text_secrets
from agentqa.workbench.campaign_store import CampaignStore
from agentqa.workbench.catalog import ArtifactCatalogService
from agentqa.workbench.generation_store import GenerationWorkbenchStore
from agentqa.workbench.models import (
    ArtifactRef,
    Campaign,
    CatalogArtifact,
    FailureEvidencePreview,
    FailureInboxItem,
    HandoffBundle,
    WorkbenchProject,
    WorkflowGateResult,
    WorkflowReport,
    WorkflowStep,
)
from agentqa.workbench.orchestrator import WorkflowOrchestrator
from agentqa.workbench.preflight import ProjectPreflightService
from agentqa.workbench.store import WorkflowStore


class AssuranceWorkbenchError(AgentQAError):
    def __init__(self, message: str, *, code: str = "ASSURANCE_WORKBENCH_ERROR") -> None:
        super().__init__(message, code=code)


def _atomic_write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        digest = sha256_digest(payload)
        checksum = path.with_name(f"{path.name}.sha256")
        checksum.write_text(f"{digest}\n", encoding="utf-8")
        return digest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _artifact_ref(artifact: CatalogArtifact) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
        uri=artifact.path,
        schema_version=artifact.schema_version,
        checksum=artifact.checksum,
        source="workflow",
    )


class AssuranceWorkbenchService:
    """Orchestrate existing evidence, analysis, compare, and gate primitives."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        artifact_directory: Path,
        result_store: ResultStore,
        workflow_store: WorkflowStore,
        campaign_store: CampaignStore,
        generation_store: GenerationWorkbenchStore,
        artifact_catalog: ArtifactCatalogService,
        orchestrator: WorkflowOrchestrator,
        preflight: ProjectPreflightService,
        provider: StructuredLLMProvider | None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.artifact_directory = artifact_directory.resolve()
        self.result_store = result_store
        self.workflow_store = workflow_store
        self.campaign_store = campaign_store
        self.generation_store = generation_store
        self.artifact_catalog = artifact_catalog
        self.orchestrator = orchestrator
        self.preflight = preflight
        self.provider = provider

    async def _context(self, workflow_id: str) -> tuple[Any, WorkbenchProject]:
        workflow = await self.workflow_store.get_workflow(workflow_id)
        if workflow is None:
            raise AssuranceWorkbenchError("workflow does not exist", code="WORKFLOW_NOT_FOUND")
        project = await self.workflow_store.get_project(workflow.project_id)
        if project is None:
            raise AssuranceWorkbenchError(
                "project does not exist", code="WORKBENCH_PROJECT_NOT_FOUND"
            )
        return workflow, project

    def _path(self, relative: str) -> Path:
        candidate = (self.workspace_root / PurePosixPath(relative)).resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError:
            raise AssuranceWorkbenchError(
                "artifact path escapes the workspace", code="ASSURANCE_ARTIFACT_PATH_ERROR"
            ) from None
        return candidate

    async def _analysis_reports(
        self, project_id: str
    ) -> list[tuple[CatalogArtifact, AnalysisReport]]:
        reports: list[tuple[CatalogArtifact, AnalysisReport]] = []
        for artifact in await self.artifact_catalog.list_artifacts(project_id, "AnalysisReport"):
            if artifact.availability != "AVAILABLE":
                continue
            try:
                reports.append(
                    (
                        artifact,
                        AnalysisReport.model_validate_json(self._path(artifact.path).read_bytes()),
                    )
                )
            except (OSError, ValidationError):
                continue
        return reports

    async def _failure_source(
        self, workflow_id: str, failure_id: str
    ) -> tuple[Campaign, Any, Any, Any, WorkbenchProject]:
        _, project = await self._context(workflow_id)
        for campaign in await self.campaign_store.list(workflow_id):
            risk_by_case = {item.case_id: item.risk for item in campaign.config.tests}
            revision_by_case = {item.case_id: item.spec_hash for item in campaign.config.tests}
            for subrun in campaign.subruns:
                stored = await self.result_store.get_run(subrun.run_id)
                if stored is None or stored.artifact_path is None:
                    continue
                for result in await self.result_store.list_results(subrun.run_id):
                    if result.passed:
                        continue
                    identity = sha256_digest(
                        canonical_json_bytes(
                            {
                                "campaign_id": campaign.campaign_id,
                                "run_id": subrun.run_id,
                                "trial_id": result.trial_id,
                            }
                        )
                    )
                    if identity == failure_id:
                        inputs = load_failure_analysis_input(
                            Path(stored.artifact_path), result.trial_id
                        )
                        return (
                            campaign,
                            subrun,
                            inputs,
                            {
                                "risk": risk_by_case.get(result.test_case_id, "low"),
                                "revision": revision_by_case.get(
                                    result.test_case_id, hash_spec(inputs.test_spec)
                                ),
                            },
                            project,
                        )
        raise AssuranceWorkbenchError("failure does not exist", code="FAILURE_NOT_FOUND")

    async def failures(
        self,
        workflow_id: str,
        *,
        risk: str | None = None,
        evaluator: str | None = None,
        tool: str | None = None,
        mutation: str | None = None,
        target: str | None = None,
        test_revision: str | None = None,
        analysis_status: str | None = None,
    ) -> list[FailureInboxItem]:
        workflow, project = await self._context(workflow_id)
        reports = await self._analysis_reports(project.project_id)
        items: list[FailureInboxItem] = []
        running = {
            cast(str, step.input_data.get("failure_id")): step.status
            for step in workflow.steps
            if step.kind == "ANALYZE_FAILURE"
            and step.status in {"QUEUED", "RUNNING", "FAILED_SAFE"}
        }
        for campaign in await self.campaign_store.list(workflow_id):
            risk_by_case = {item.case_id: item.risk for item in campaign.config.tests}
            revision_by_case = {item.case_id: item.spec_hash for item in campaign.config.tests}
            for subrun in campaign.subruns:
                for result in await self.result_store.list_results(subrun.run_id):
                    if result.passed:
                        continue
                    failure_id = sha256_digest(
                        canonical_json_bytes(
                            {
                                "campaign_id": campaign.campaign_id,
                                "run_id": subrun.run_id,
                                "trial_id": result.trial_id,
                            }
                        )
                    )
                    matches = [
                        (ref, report)
                        for ref, report in reports
                        if report.run_id == subrun.run_id and report.trial_id == result.trial_id
                    ]
                    latest = max(matches, key=lambda pair: pair[1].generated_at, default=None)
                    state = running.get(failure_id)
                    resolved_status = (
                        "ANALYZING"
                        if state in {"QUEUED", "RUNNING"}
                        else "FAILED_SAFE"
                        if state == "FAILED_SAFE"
                        else "ANALYZED"
                        if matches
                        else "NOT_ANALYZED"
                    )
                    try:
                        trace = await self.result_store.get_trace(result.trace_id)
                        tools = (
                            sorted({event.name for event in trace.events if event.kind == "TOOL"})
                            if trace is not None
                            else []
                        )
                    except Exception:
                        tools = []
                    item = FailureInboxItem(
                        failure_id=failure_id,
                        workflow_id=workflow_id,
                        campaign_id=campaign.campaign_id,
                        run_id=subrun.run_id,
                        target_role=subrun.target.role,
                        target_name=subrun.target.name,
                        test_case_id=result.test_case_id,
                        trial_id=result.trial_id,
                        trace_id=result.trace_id,
                        risk=risk_by_case.get(result.test_case_id, "low"),
                        evaluator_statuses={e.evaluator_type: e.status for e in result.evaluations},
                        failed_evaluators=[
                            e.evaluator_type for e in result.evaluations if e.status != "PASS"
                        ],
                        tools=tools,
                        mutation_ids=sorted({m.mutation_id for m in result.mutation_applications}),
                        test_revision_hash=revision_by_case.get(
                            result.test_case_id, sha256_digest(result.test_case_id.encode())
                        ),
                        failure_bundle_ref=result.failure_bundle_ref,
                        trace_status=result.trace_status,
                        analysis_status=cast(Any, resolved_status),
                        analysis_count=len(matches),
                        latest_analysis_id=None if latest is None else latest[1].analysis_id,
                        latest_analysis_ref=None if latest is None else latest[0].path,
                    )
                    if risk is not None and item.risk != risk:
                        continue
                    if evaluator is not None and evaluator not in item.failed_evaluators:
                        continue
                    if tool is not None and tool not in item.tools:
                        continue
                    if mutation is not None and mutation not in item.mutation_ids:
                        continue
                    if target is not None and target not in {item.target_role, item.target_name}:
                        continue
                    if test_revision is not None and item.test_revision_hash != test_revision:
                        continue
                    if analysis_status is not None and item.analysis_status != analysis_status:
                        continue
                    items.append(item)
        return sorted(items, key=lambda item: (item.risk, item.run_id, item.trial_id), reverse=True)

    async def preview(
        self, workflow_id: str, failure_id: str, code_evidence_hashes: Sequence[str]
    ) -> FailureEvidencePreview:
        campaign, subrun, inputs, metadata, project = await self._failure_source(
            workflow_id, failure_id
        )
        del campaign, subrun, metadata
        available = {
            item.content_hash: item
            for item in await self.artifact_catalog.list_code_evidence(project.project_id)
        }
        unknown = sorted(set(code_evidence_hashes) - set(available))
        if unknown:
            raise AssuranceWorkbenchError(
                "selected code evidence is not registered for this project",
                code="ANALYSIS_CODE_EVIDENCE_NOT_FOUND",
            )
        selected: list[CodeEvidence] = [available[value] for value in code_evidence_hashes]
        bundle = build_evidence_bundle(
            run_id=inputs.run_id,
            trial_id=inputs.result.trial_id,
            test_spec=inputs.test_spec,
            target_result=inputs.result.target_result,
            trace_record=inputs.trace,
            manifest=inputs.manifest,
            evaluations=inputs.result.evaluations,
            code_evidence=selected,
        )
        failures = await self.failures(workflow_id)
        failure = next(item for item in failures if item.failure_id == failure_id)
        digest = evidence_bundle_hash(bundle)
        reusable = [
            report
            for _, report in await self._analysis_reports(project.project_id)
            if report.run_id == inputs.run_id
            and report.trial_id == inputs.result.trial_id
            and report.evidence_hash == digest
        ]
        return FailureEvidencePreview(
            failure=failure, evidence_hash=digest, bundle=bundle, reusable_analyses=reusable
        )

    async def analyze(
        self,
        workflow_id: str,
        failure_id: str,
        *,
        idempotency_key: str,
        code_evidence_hashes: Sequence[str],
        reuse_existing: bool,
    ) -> dict[str, Any]:
        if self.provider is None:
            raise AssuranceWorkbenchError(
                "AI analysis requires a configured provider", code="ANALYSIS_PROVIDER_REQUIRED"
            )
        preview = await self.preview(workflow_id, failure_id, code_evidence_hashes)
        if reuse_existing and preview.reusable_analyses:
            report = max(preview.reusable_analyses, key=lambda item: item.generated_at)
            await self.workflow_store.record_event(
                workflow_id,
                event_type="ANALYSIS_REUSED",
                message=f"Reused analysis {report.analysis_id} for identical evidence.",
                data={
                    "failure_id": failure_id,
                    "analysis_id": report.analysis_id,
                    "evidence_hash": report.evidence_hash,
                },
            )
            return {"reused": True, "report": report, "step": None}
        _, _, inputs, _, project = await self._failure_source(workflow_id, failure_id)
        available = {
            item.content_hash: item
            for item in await self.artifact_catalog.list_code_evidence(project.project_id)
        }
        selected = [available[value] for value in code_evidence_hashes]

        async def handler(step: WorkflowStep) -> list[ArtifactRef]:
            report = await FailureAnalysisService(
                cast(StructuredLLMProvider, self.provider)
            ).analyze(
                run_id=inputs.run_id,
                test_spec=inputs.test_spec,
                result=inputs.result,
                trace=inputs.trace,
                manifest=inputs.manifest,
                code_evidence=selected,
            )
            destination = write_analysis_report(
                report,
                self.artifact_directory
                / "workflows"
                / workflow_id
                / "analyses"
                / f"analysis-{report.analysis_id}.json",
                workspace=self.workspace_root,
            )
            artifact = await self.artifact_catalog.register_path(
                project.project_id, destination, source="WORKFLOW", workflow_id=workflow_id
            )
            return [_artifact_ref(artifact)]

        step = await self.orchestrator.submit(
            workflow_id=workflow_id,
            kind="ANALYZE_FAILURE",
            idempotency_key=idempotency_key,
            input_data={
                "failure_id": failure_id,
                "run_id": inputs.run_id,
                "trial_id": inputs.result.trial_id,
                "evidence_hash": preview.evidence_hash,
                "code_evidence_hashes": list(code_evidence_hashes),
            },
            handler=handler,
        )
        return {"reused": False, "report": None, "step": step}

    async def analyses(self, workflow_id: str) -> list[AnalysisReport]:
        _, project = await self._context(workflow_id)
        reports = await self._analysis_reports(project.project_id)
        campaign_run_ids = {
            subrun.run_id
            for campaign in await self.campaign_store.list(workflow_id)
            for subrun in campaign.subruns
        }
        return sorted(
            [
                report
                for artifact, report in reports
                if artifact.workflow_id == workflow_id or report.run_id in campaign_run_ids
            ],
            key=lambda item: item.generated_at,
            reverse=True,
        )

    async def _analysis(
        self, workflow_id: str, analysis_id: str
    ) -> tuple[CatalogArtifact, AnalysisReport]:
        _, project = await self._context(workflow_id)
        for artifact, report in await self._analysis_reports(project.project_id):
            if report.analysis_id == analysis_id:
                return artifact, report
        raise AssuranceWorkbenchError("analysis does not exist", code="ANALYSIS_NOT_FOUND")

    async def create_handoff(
        self, workflow_id: str, analysis_id: str, developer_task: str
    ) -> tuple[HandoffBundle, ArtifactRef]:
        workflow, project = await self._context(workflow_id)
        if workflow.status != "READY":
            raise AssuranceWorkbenchError(
                "workflow must be READY to request external repair",
                code="WORKFLOW_TRANSITION_ERROR",
            )
        analysis_artifact, analysis = await self._analysis(workflow_id, analysis_id)
        campaigns = await self.campaign_store.list(workflow_id)
        selected = next(
            (
                (campaign, subrun)
                for campaign in campaigns
                for subrun in campaign.subruns
                if subrun.run_id == analysis.run_id
            ),
            None,
        )
        if selected is None:
            raise AssuranceWorkbenchError(
                "analysis is not attached to this workflow campaign",
                code="ANALYSIS_WORKFLOW_MISMATCH",
            )
        campaign, subrun = selected
        snapshot = campaign.config.manifest_snapshot
        handoff = HandoffBundle(
            handoff_id=new_id(),
            created_at=utc_now(),
            workflow_id=workflow_id,
            project_id=project.project_id,
            failure_resource_uri=f"agentqa://runs/{analysis.run_id}/failures/{analysis.trial_id}",
            analysis_resource_uri=f"agentqa://analyses/{analysis.analysis_id}",
            baseline_target_hash=subrun.target.spec_hash,
            baseline_manifest_hash=campaign.config.manifest_hash,
            baseline_source_snapshot_hash=sha256_digest(canonical_json_bytes(snapshot.source)),
            baseline_runtime_hash=sha256_digest(canonical_json_bytes(snapshot.runtime_contract)),
            suggested_regression_tests=[item.title for item in analysis.regression_tests],
            missing_evidence=analysis.missing_evidence,
            developer_task=redact_text_secrets(developer_task),
        )
        path = (
            self.artifact_directory
            / "workflows"
            / workflow_id
            / "handoffs"
            / handoff.handoff_id
            / "handoff-bundle.json"
        )
        checksum = _atomic_write(path, canonical_json_bytes(handoff) + b"\n")
        artifact = await self.artifact_catalog.register_path(
            project.project_id, path, source="WORKFLOW", workflow_id=workflow_id
        )
        ref = _artifact_ref(artifact).model_copy(
            update={
                "artifact_type": "HandoffBundle",
                "schema_version": handoff.schema_version,
                "checksum": checksum,
            }
        )
        await self.workflow_store.record_event(
            workflow_id,
            event_type="HANDOFF_BUNDLE_CREATED",
            message=f"Created bounded handoff {handoff.handoff_id} from analysis {analysis_id}.",
            data={
                "handoff_id": handoff.handoff_id,
                "analysis_ref": analysis_artifact.path,
                "checksum": checksum,
            },
        )
        await self.workflow_store.wait_for_external_repair(
            workflow_id, handoff_id=handoff.handoff_id, checksum=checksum
        )
        return handoff, ref

    async def rerun_preflight(self, workflow_id: str) -> dict[str, Any]:
        _, project = await self._context(workflow_id)
        before = project.last_manifest_snapshot
        report = await self.preflight.run(project)
        after = report.manifest_snapshot
        return {
            "report": report,
            "changes": {
                "manifest": None
                if before is None or after is None
                else before.declared_manifest_hash != after.declared_manifest_hash,
                "source": None
                if before is None or after is None
                else sha256_digest(canonical_json_bytes(before.source))
                != sha256_digest(canonical_json_bytes(after.source)),
                "runtime": None
                if before is None or after is None
                else sha256_digest(canonical_json_bytes(before.runtime_contract))
                != sha256_digest(canonical_json_bytes(after.runtime_contract)),
            },
        }

    async def compare(
        self,
        workflow_id: str,
        baseline_run_id: str,
        candidate_run_id: str,
        idempotency_key: str,
    ) -> tuple[ComparisonReport, WorkflowStep]:
        _, project = await self._context(workflow_id)

        async def handler(step: WorkflowStep) -> list[ArtifactRef]:
            baseline = await self._run_artifact(baseline_run_id)
            candidate = await self._run_artifact(candidate_run_id)
            report = compare_runs(load_run_artifact(baseline), load_run_artifact(candidate))
            path = (
                self.artifact_directory
                / "workflows"
                / workflow_id
                / "comparisons"
                / step.step_id
                / "comparison.json"
            )
            write_report(path, report)
            artifact = await self.artifact_catalog.register_path(
                project.project_id, path, source="WORKFLOW", workflow_id=workflow_id
            )
            return [_artifact_ref(artifact)]

        step = await self.orchestrator.submit(
            workflow_id=workflow_id,
            kind="COMPARE_RUNS",
            idempotency_key=idempotency_key,
            input_data={"baseline_run_id": baseline_run_id, "candidate_run_id": candidate_run_id},
            handler=handler,
        )
        finished = await self.orchestrator.wait(step.step_id)
        if finished is None or finished.status != "COMPLETED" or not finished.artifacts:
            raise AssuranceWorkbenchError(
                "comparison failed safely", code="COMPARISON_WORKFLOW_FAILED"
            )
        report = ComparisonReport.model_validate_json(
            self._path(finished.artifacts[0].uri).read_bytes()
        )
        return report, finished

    async def _run_artifact(self, run_id: str) -> Path:
        stored = await self.result_store.get_run(run_id)
        if stored is None or stored.artifact_path is None:
            raise AssuranceWorkbenchError(
                "run artifact is unavailable", code="RUN_ARTIFACT_UNAVAILABLE"
            )
        return Path(stored.artifact_path)

    def _gate_spec(self, project: WorkbenchProject, gate_path: str) -> QualityGateSpec:
        root = (self.workspace_root / PurePosixPath(project.workspace_path)).resolve()
        path = (root / PurePosixPath(gate_path)).resolve()
        try:
            path.relative_to(root)
            loaded = load_spec_file(path)
        except (ValueError, OSError, SpecLoadError):
            raise AssuranceWorkbenchError(
                "gate specification is invalid", code="GATE_SPEC_ERROR"
            ) from None
        specs = [item.spec for item in loaded if isinstance(item.spec, QualityGateSpec)]
        if len(loaded) != 1 or len(specs) != 1:
            raise AssuranceWorkbenchError(
                "gate file must contain one QualityGate", code="GATE_SPEC_ERROR"
            )
        return specs[0]

    async def gate(
        self,
        workflow_id: str,
        *,
        kind: Literal["regression", "mutation"],
        gate_path: str,
        candidate_run_id: str,
        baseline_run_id: str | None,
        idempotency_key: str,
    ) -> tuple[WorkflowGateResult, WorkflowStep]:
        _, project = await self._context(workflow_id)
        spec = self._gate_spec(project, gate_path)
        if kind == "regression" and spec.rules.min_mutation_score is not None:
            raise AssuranceWorkbenchError(
                "regression gate must not include minMutationScore; run a separate mutation gate",
                code="GATE_KIND_MISMATCH",
            )
        if kind == "mutation" and any(
            value is not None
            for value in (
                spec.rules.max_new_regressions,
                spec.rules.max_p95_latency_increase,
                spec.rules.max_average_cost_increase,
            )
        ):
            raise AssuranceWorkbenchError(
                "mutation gate must not include regression-relative rules",
                code="GATE_KIND_MISMATCH",
            )

        async def handler(step: WorkflowStep) -> list[ArtifactRef]:
            candidate = load_run_artifact(await self._run_artifact(candidate_run_id))
            comparison = None
            if baseline_run_id is not None:
                comparison = compare_runs(
                    load_run_artifact(await self._run_artifact(baseline_run_id)), candidate
                )
            result = evaluate_gate(spec, candidate, comparison)
            path = (
                self.artifact_directory
                / "workflows"
                / workflow_id
                / "gates"
                / step.step_id
                / f"{kind}-gate.json"
            )
            write_report(path, result)
            artifact = await self.artifact_catalog.register_path(
                project.project_id, path, source="WORKFLOW", workflow_id=workflow_id
            )
            return [_artifact_ref(artifact)]

        step = await self.orchestrator.submit(
            workflow_id=workflow_id,
            kind="EVALUATE_GATE",
            idempotency_key=idempotency_key,
            input_data={
                "kind": kind,
                "gate_path": gate_path,
                "gate_spec_hash": hash_spec(spec),
                "candidate_run_id": candidate_run_id,
                "baseline_run_id": baseline_run_id,
            },
            handler=handler,
        )
        finished = await self.orchestrator.wait(step.step_id)
        if finished is None or finished.status != "COMPLETED" or not finished.artifacts:
            raise AssuranceWorkbenchError("gate failed safely", code="GATE_WORKFLOW_FAILED")
        result = GateResult.model_validate_json(self._path(finished.artifacts[0].uri).read_bytes())
        record = WorkflowGateResult(
            gate_id=finished.step_id,
            workflow_id=workflow_id,
            created_at=utc_now(),
            kind=kind,
            gate_path=gate_path,
            gate_spec_hash=hash_spec(spec),
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            result=result,
        )
        return record, finished

    async def export_report(
        self, workflow_id: str, idempotency_key: str
    ) -> tuple[WorkflowReport, WorkflowStep]:
        workflow, project = await self._context(workflow_id)

        async def handler(step: WorkflowStep) -> list[ArtifactRef]:
            campaigns = await self.campaign_store.list(workflow_id)
            sessions = await self.generation_store.list_sessions(workflow_id)
            analyses = await self.analyses(workflow_id)
            approvals = await self.workflow_store.list_approvals(workflow_id)
            run_ids = [subrun.run_id for campaign in campaigns for subrun in campaign.subruns]
            run_id_set = set(run_ids)
            artifacts = [
                item
                for item in await self.artifact_catalog.list_artifacts(project.project_id)
                if (item.workflow_id == workflow_id or item.run_id in run_id_set)
                and item.availability == "AVAILABLE"
            ]
            stored_runs = [await self.result_store.get_run(run_id) for run_id in run_ids]
            total = sum(run.run.summary.total for run in stored_runs if run is not None)
            passed = sum(run.run.summary.passed for run in stored_runs if run is not None)
            missing: list[str] = []
            token_values = [
                subrun.budget.consumed_total_tokens
                for campaign in campaigns
                for subrun in campaign.subruns
                if subrun.budget is not None and subrun.budget.token_observations > 0
            ]
            cost_values = [
                subrun.budget.consumed_cost_usd
                for campaign in campaigns
                for subrun in campaign.subruns
                if subrun.budget is not None and subrun.budget.cost_observations > 0
            ]
            if not token_values:
                missing.append("token usage was not observable")
            if not cost_values:
                missing.append("cost was not observable")
            latency_values = [
                case.latency_ms.mean
                for run in stored_runs
                if run is not None
                for case in run.run.summary.cases
                if case.latency_ms.mean is not None
            ]
            if not latency_values:
                missing.append("latency was not observable")
            missing.extend(gap for report in analyses for gap in report.missing_evidence)
            snapshots = [campaign.config.manifest_snapshot for campaign in campaigns]
            providers = [call for session in sessions for call in session.llm_calls] + [
                report.llm_call for report in analyses
            ]
            generation_reports = []
            for session in sessions:
                if session.report_path is None:
                    continue
                try:
                    generation_reports.append(
                        GenerationReport.model_validate_json(
                            self._path(session.report_path).read_bytes()
                        )
                    )
                except (OSError, ValidationError):
                    missing.append(
                        f"generation report for session {session.session_id} is unavailable"
                    )
            comparison_docs: list[dict[str, JsonValue]] = []
            gate_docs: list[dict[str, JsonValue]] = []
            for artifact in artifacts:
                if artifact.artifact_type not in {"ComparisonReport", "GateResult"}:
                    continue
                try:
                    raw = cast(
                        dict[str, JsonValue], json.loads(self._path(artifact.path).read_bytes())
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    continue
                if artifact.artifact_type == "ComparisonReport":
                    comparison_docs.append(raw)
                elif artifact.artifact_type == "GateResult":
                    gate_docs.append({"artifact": artifact.path, "result": raw})
            refs = [_artifact_ref(item) for item in artifacts]
            try:
                report = WorkflowReport(
                    report_id=new_id(),
                    generated_at=utc_now(),
                    workflow_id=workflow_id,
                    project={
                        "project_id": project.project_id,
                        "name": project.name,
                        "workspace_path": project.workspace_path,
                    },
                    identity=cast(
                        dict[str, JsonValue],
                        {
                            "manifest_hashes": sorted({c.config.manifest_hash for c in campaigns}),
                            "runtime_hashes": sorted(
                                {
                                    sha256_digest(canonical_json_bytes(s.runtime_contract))
                                    for s in snapshots
                                }
                            ),
                            "source_snapshot_hashes": sorted(
                                {sha256_digest(canonical_json_bytes(s.source)) for s in snapshots}
                            ),
                            "target_hashes": sorted(
                                {subrun.target.spec_hash for c in campaigns for subrun in c.subruns}
                            ),
                        },
                    ),
                    provider={
                        "calls": [
                            {
                                "use_case": call.use_case,
                                "provider": call.provider,
                                "model": call.model,
                                "parameters": call.parameters,
                                "status": call.status,
                            }
                            for call in providers
                        ]
                    },
                    generation={
                        "sessions": len(sessions),
                        "candidate_ready": sum(s.status == "CANDIDATES_READY" for s in sessions),
                        "generated_cases": sum(len(item.cases) for item in generation_reports),
                        "valid_cases": sum(len(item.cases) for item in generation_reports),
                        "valid_rate": 1.0 if generation_reports else None,
                        "coverage_gaps": sum(len(item.gaps) for item in generation_reports),
                        "approved": sum(
                            a.decision == "APPROVED" and a.subject_kind == "GeneratedTest"
                            for a in approvals
                        ),
                        "edited": sum(
                            1
                            for event in await self.workflow_store.list_events(workflow_id)
                            if event.event_type == "GENERATED_TEST_REVISED"
                        ),
                    },
                    test_results={
                        "campaigns": len(campaigns),
                        "runs": len(run_ids),
                        "total_cases": total,
                        "passed_cases": passed,
                        "pass_rate": None if total == 0 else passed / total,
                        "mutation_scores": [
                            None
                            if run.run.mutation_score is None
                            else run.run.mutation_score.overall.score
                            for run in stored_runs
                            if run is not None
                        ],
                        "coverage": None,
                    },
                    analyses={
                        "total": len(analyses),
                        "reports": [
                            {
                                "analysis_id": a.analysis_id,
                                "run_id": a.run_id,
                                "trial_id": a.trial_id,
                                "evidence_hash": a.evidence_hash,
                                "status": a.llm_call.status,
                            }
                            for a in analyses
                        ],
                    },
                    comparisons=comparison_docs,
                    gates=gate_docs,
                    resources={
                        "tokens": None if not token_values else sum(token_values),
                        "cost_usd": None if not cost_values else sum(cost_values),
                        "latency_ms_mean": (
                            None
                            if not latency_values
                            else sum(latency_values) / len(latency_values)
                        ),
                        "human_approvals": len(approvals),
                    },
                    approvals={
                        "count": len(approvals),
                        "decisions": [a.model_dump(mode="json") for a in approvals],
                    },
                    missing_evidence=list(dict.fromkeys(missing)),
                    authoritative_artifacts=refs,
                )
            except ValidationError as exc:
                issue = exc.errors(include_url=False)[0]
                raise AssuranceWorkbenchError(
                    f"workflow report could not be validated: {issue['msg']}",
                    code="REPORT_FORMAT_ERROR",
                ) from None
            path = (
                self.artifact_directory
                / "workflows"
                / workflow_id
                / "reports"
                / step.step_id
                / "workflow-report.json"
            )
            _atomic_write(path, canonical_json_bytes(report) + b"\n")
            artifact = await self.artifact_catalog.register_path(
                project.project_id, path, source="WORKFLOW", workflow_id=workflow_id
            )
            return [_artifact_ref(artifact)]

        step = await self.orchestrator.submit(
            workflow_id=workflow_id,
            kind="EXPORT_REPORT",
            idempotency_key=idempotency_key,
            input_data={"workflow_revision": workflow.revision},
            handler=handler,
        )
        finished = await self.orchestrator.wait(step.step_id)
        if finished is None or finished.status != "COMPLETED" or not finished.artifacts:
            message = (
                "report export failed safely"
                if finished is None or finished.error is None
                else finished.error.message
            )
            raise AssuranceWorkbenchError(message, code="REPORT_WORKFLOW_FAILED")
        report = WorkflowReport.model_validate_json(
            self._path(finished.artifacts[0].uri).read_bytes()
        )
        return report, finished


__all__ = ["AssuranceWorkbenchError", "AssuranceWorkbenchService"]
