import asyncio
import hashlib
import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from pydantic import JsonValue

from agentqa.api import ApiSettings, create_app
from agentqa.evaluators import JudgeRequest
from agentqa.llm import StructuredRequest, StructuredResponse
from agentqa.models import ErrorInfo
from agentqa.storage import FileResultStore
from agentqa.workbench.store import SQLiteWorkflowStore
from tests.fixtures.stored_runs import create_storable_run

pytestmark = pytest.mark.integration


class _PassJudgeProvider:
    name = "fake-deepseek"
    model = "fake-model"
    parameters: Mapping[str, JsonValue] = {"temperature": 0.0}

    async def judge(self, request: JudgeRequest) -> Mapping[str, JsonValue]:
        assert "Return JSON" in request.prompt
        return {
            "verdict": "PASS",
            "score": 1.0,
            "rationale": "The response satisfies the rubric.",
            "violations": [],
        }


class _AnalysisProvider(_PassJudgeProvider):
    async def complete(self, request: StructuredRequest) -> StructuredResponse:
        assert request.use_case == "failure_analysis"
        return StructuredResponse(
            content={
                "observed_facts": [
                    {
                        "statement": "The normalized target output did not satisfy the test.",
                        "evidence_ids": ["target-result", "evaluation-01"],
                    }
                ],
                "earliest_divergence": {
                    "description": "The returned answer first diverged at target output.",
                    "evidence_ids": ["scenario", "target-result"],
                    "span_id": None,
                },
                "root_causes": [
                    {
                        "id": "wrong-answer",
                        "hypothesis": "The target echoed input instead of the required answer.",
                        "confidence": 0.7,
                        "evidence_ids": ["scenario", "target-result"],
                        "counter_evidence_ids": [],
                        "reasoning": (
                            "Scenario and output disagree with the deterministic evaluator."
                        ),
                    }
                ],
                "additional_evidence_needed": ["A complete internal trajectory is unavailable."],
                "regression_tests": [
                    {
                        "title": "Keep the required answer",
                        "rationale": "Protect the observed behavior boundary.",
                        "expected_behavior": "Return the declared exact answer.",
                        "evidence_ids": ["scenario", "target-result"],
                    }
                ],
                "repair_directions": ["Inspect the answer mapping before changing the target."],
            },
            metadata={"request_id": "analysis-1", "usage": {"total_tokens": 50}},
        )


def _settings(tmp_path: Path, workspace: Path | None = None) -> ApiSettings:
    return ApiSettings(
        workspace_root=(workspace or tmp_path),
        artifact_directory=tmp_path / "api-artifacts",
        database_path=tmp_path / "unused.sqlite3",
        static_directory=None,
        max_receiver_bytes=1024,
    )


def test_api_lists_run_details_results_traces_findings_and_reviews(
    tmp_path: Path,
) -> None:
    run, artifacts = create_storable_run(tmp_path / "artifacts", "api-run")
    store = FileResultStore(tmp_path / "store")
    asyncio.run(store.save_run(run, project_id="campus", artifact_path=artifacts))
    app = create_app(_settings(tmp_path), store=store)

    with TestClient(app) as client:
        health = client.get("/api/health")
        ai_evidence = client.get("/api/ai/evidence")
        listing = client.get("/api/runs", params={"project_id": "campus"})
        detail = client.get(f"/api/runs/{run.run_id}")
        results = client.get(f"/api/runs/{run.run_id}/results")
        trace = client.get(f"/api/traces/{run.results[0].trace_id}")
        findings = client.get(f"/api/runs/{run.run_id}/findings")

        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert ai_evidence.json() == {"generations": [], "analyses": []}
        assert listing.json()["runs"][0]["run_id"] == run.run_id
        assert detail.json()["state"] == "stored"
        assert results.json()["results"][0]["trial_id"] == run.results[0].trial_id
        assert trace.json()["status"] == "missing"
        finding_id = findings.json()["findings"][0]["finding_id"]

        reviewed = client.put(
            f"/api/findings/{finding_id}/review",
            json={"status": "confirmed", "note": "checked"},
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["status"] == "confirmed"


def test_api_compares_runs_and_evaluates_gate(tmp_path: Path) -> None:
    baseline, baseline_artifacts = create_storable_run(
        tmp_path / "baseline-artifacts", "baseline-run", passed=True
    )
    candidate, candidate_artifacts = create_storable_run(
        tmp_path / "candidate-artifacts", "candidate-run", passed=False
    )
    gate = tmp_path / "gate.yaml"
    gate.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: QualityGate
metadata:
  name: api-gate
rules:
  minPassRate: 1.0
  maxNewRegressions: 0
""",
        encoding="utf-8",
    )
    store = FileResultStore(tmp_path / "store")
    asyncio.run(store.save_run(baseline, artifact_path=baseline_artifacts))
    asyncio.run(store.save_run(candidate, artifact_path=candidate_artifacts))
    app = create_app(_settings(tmp_path), store=store)

    with TestClient(app) as client:
        comparison = client.post(
            "/api/comparisons",
            json={
                "baselineRunId": baseline.run_id,
                "candidateRunId": candidate.run_id,
            },
        )
        gated = client.post(
            "/api/gates/evaluate",
            json={
                "gatePath": "gate.yaml",
                "baselineRunId": baseline.run_id,
                "candidateRunId": candidate.run_id,
            },
        )

    assert comparison.status_code == 200
    assert comparison.json()["counts"]["new_regressions"] == 1
    assert gated.status_code == 200
    assert gated.json()["passed"] is False
    assert gated.json()["exit_code"] == 1


def test_api_persists_workbench_workflows_and_event_cursors(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), store=FileResultStore(tmp_path / "store"))

    with TestClient(app) as client:
        project = client.post(
            "/api/projects",
            json={
                "projectId": "demo",
                "name": "Demo agent",
                "description": "Local workbench project",
            },
        )
        assert project.status_code == 201
        assert project.json()["project_id"] == "demo"

        request = {
            "projectId": "demo",
            "name": "First assurance workflow",
            "idempotencyKey": "demo-workflow-1",
            "draft": True,
        }
        created = client.post("/api/workflows", json=request)
        repeated = client.post("/api/workflows", json=request)
        assert created.status_code == 201
        assert repeated.json()["workflow_id"] == created.json()["workflow_id"]
        workflow_id = created.json()["workflow_id"]

        resumed = client.post(f"/api/workflows/{workflow_id}/resume")
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "READY"

        first_page = client.get(f"/api/workflows/{workflow_id}/events", params={"after": 0})
        assert [event["revision"] for event in first_page.json()["events"]] == [1, 2]
        assert first_page.json()["revision"] == 2
        assert (
            client.get(f"/api/workflows/{workflow_id}/events", params={"after": 2}).json()["events"]
            == []
        )

        cancelled = client.post(f"/api/workflows/{workflow_id}/cancel")
        listing = client.get("/api/workflows", params={"project_id": "demo"})
        assert cancelled.json()["status"] == "CANCELLED"
        assert listing.json()["workflows"][0]["workflow_id"] == workflow_id

        conflict = client.post(
            "/api/workflows",
            json={**request, "name": "Changed workflow"},
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "WORKFLOW_IDEMPOTENCY_CONFLICT"


def test_api_deletes_project_metadata_but_preserves_workspace_files(tmp_path: Path) -> None:
    marker = tmp_path / "agent-source.py"
    marker.write_text("print('kept')\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        created = client.post(
            "/api/projects", json={"projectId": "remove-me", "name": "Old project"}
        )
        assert created.status_code == 201
        workflow = client.post(
            "/api/workflows",
            json={"projectId": "remove-me", "name": "Old evaluation", "idempotencyKey": "old-1"},
        )
        assert workflow.status_code == 201

        deleted = client.delete("/api/projects/remove-me")
        assert deleted.status_code == 204
        assert client.get("/api/projects/remove-me").status_code == 404
        assert client.get(f"/api/workflows/{workflow.json()['workflow_id']}").status_code == 404
        assert marker.read_text(encoding="utf-8") == "print('kept')\n"


def test_campaign_builder_confirms_exact_config_and_persists_target_run(
    tmp_path: Path,
) -> None:
    (tmp_path / "demo_agent.py").write_text(
        "def run(payload):\n    return {'answer': payload.get('input', {}).get('message', '')}\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha2
kind: AgentManifest
metadata:
  name: campaign-agent
runtime:
  language: python
  framework: custom
  entrypoint: demo_agent:run
purpose:
  summary: Echo a requested message.
capabilities:
  - name: echo
    description: Echo one message.
tools:
  - name: external_writer
    description: Demonstrate an externally visible operation.
    sideEffect: external-write
    risk: high
    idempotent: true
policies:
  forbiddenBehaviors:
    - Do not alter the selected campaign configuration.
quality:
  objectives: [task-success]
""",
        encoding="utf-8",
    )
    (tmp_path / "target.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: demo-target
type: python
entrypoint: demo_agent:run
workingDirectory: .
""",
        encoding="utf-8",
    )
    (tmp_path / "test.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: AgentTest
metadata:
  name: echo-high-risk
  labels:
    risk: high
scenario:
  input:
    message: hello
expect:
  output:
    exactFields:
      answer: hello
evaluators:
  - type: schema
""",
        encoding="utf-8",
    )
    app = create_app(_settings(tmp_path), store=FileResultStore(tmp_path / "store"))

    with TestClient(app) as client:
        assert (
            client.post(
                "/api/projects",
                json={
                    "projectId": "campaign-demo",
                    "name": "Campaign demo",
                    "defaultSpecRefs": {
                        "manifest": "manifest.yaml",
                        "target": "target.yaml",
                        "tests": "test.yaml",
                    },
                },
            ).status_code
            == 201
        )
        assert client.post("/api/projects/campaign-demo/preflight").status_code == 200
        workflow = client.post(
            "/api/workflows",
            json={
                "projectId": "campaign-demo",
                "name": "Campaign execution",
                "idempotencyKey": "campaign-workflow",
            },
        ).json()
        workflow_id = workflow["workflow_id"]
        inputs = client.get(f"/api/workflows/{workflow_id}/campaign-inputs")
        assert inputs.status_code == 200
        assert inputs.json()["golden"][0]["caseId"] == "echo-high-risk"

        denied = client.post(
            f"/api/workflows/{workflow_id}/campaigns",
            json={
                "idempotencyKey": "denied",
                "targets": [{"role": "target", "path": "target.yaml"}],
            },
        )
        assert denied.status_code == 422
        assert denied.json()["detail"]["code"] == "CAMPAIGN_SIDE_EFFECTS_NOT_ALLOWED"

        created = client.post(
            f"/api/workflows/{workflow_id}/campaigns",
            json={
                "idempotencyKey": "campaign-1",
                "targets": [{"role": "target", "path": "target.yaml"}],
                "allowSideEffects": True,
                "replayMode": "off",
            },
        )
        assert created.status_code == 201
        campaign = created.json()
        assert campaign["status"] == "AWAITING_CONFIRMATION"
        assert campaign["risk"]["high_risk_cases"] == ["echo-high-risk"]

        stale = client.post(
            f"/api/campaigns/{campaign['campaign_id']}/start",
            json={"configHash": "sha256:" + "0" * 64, "idempotencyKey": "start-stale"},
        )
        assert stale.status_code == 409
        challenge = client.post(f"/api/campaigns/{campaign['campaign_id']}/challenge").json()
        confirmed = client.post(
            f"/api/campaigns/{campaign['campaign_id']}/confirm",
            json={
                "challengeId": challenge["challenge"]["challenge_id"],
                "phrase": challenge["phrase"],
                "configHash": campaign["config_hash"],
                "reviewer": "tester",
            },
        )
        assert confirmed.status_code == 200
        reused = client.post(
            f"/api/campaigns/{campaign['campaign_id']}/confirm",
            json={
                "challengeId": challenge["challenge"]["challenge_id"],
                "phrase": challenge["phrase"],
                "configHash": campaign["config_hash"],
                "reviewer": "tester",
            },
        )
        assert reused.status_code == 409
        started = client.post(
            f"/api/campaigns/{campaign['campaign_id']}/start",
            json={"configHash": campaign["config_hash"], "idempotencyKey": "start-1"},
        )
        assert started.status_code == 202
        final: dict[str, object] = {}
        for _ in range(150):
            final = client.get(f"/api/campaigns/{campaign['campaign_id']}").json()
            if final["status"] in {"COMPLETED", "FAILED", "INCOMPLETE"}:
                break
            time.sleep(0.02)
        assert final["status"] == "COMPLETED"
        subruns = final["subruns"]
        assert isinstance(subruns, list)
        assert len(subruns) == 1
        assert subruns[0]["completed_trials"] == 1
        run_id = subruns[0]["run_id"]
        assert client.get(f"/api/runs/{run_id}").json()["state"] == "stored"


def test_t21_failure_analysis_handoff_compare_gate_and_report(tmp_path: Path) -> None:
    (tmp_path / "demo_agent.py").write_text(
        "def run(payload):\n    return {'answer': payload.get('input', {}).get('message', '')}\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha2
kind: AgentManifest
metadata:
  name: assurance-agent
runtime:
  language: python
  framework: custom
  entrypoint: demo_agent:run
purpose:
  summary: Return a declared answer.
capabilities:
  - name: answer
    description: Return one answer.
quality:
  objectives: [task-success]
""",
        encoding="utf-8",
    )
    (tmp_path / "target.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: assurance-target
type: python
entrypoint: demo_agent:run
workingDirectory: .
""",
        encoding="utf-8",
    )
    (tmp_path / "test.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: AgentTest
metadata:
  name: wrong-answer
  labels:
    risk: medium
scenario:
  input:
    message: hello
expect:
  output:
    exactFields:
      answer: goodbye
evaluators:
  - type: schema
""",
        encoding="utf-8",
    )
    (tmp_path / "regression-gate.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: QualityGate
metadata:
  name: regression
rules:
  maxNewRegressions: 0
""",
        encoding="utf-8",
    )
    (tmp_path / "mutation-gate.yaml").write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: QualityGate
metadata:
  name: mutation
rules:
  minMutationScore: 0.5
""",
        encoding="utf-8",
    )
    workflow_store = SQLiteWorkflowStore(_settings(tmp_path).database_path)
    app = create_app(
        _settings(tmp_path),
        store=FileResultStore(tmp_path / "store"),
        workflow_store=workflow_store,
        judge_provider=_AnalysisProvider(),
    )

    with TestClient(app) as client:
        client.post(
            "/api/projects",
            json={
                "projectId": "assurance",
                "name": "Assurance project",
                "defaultSpecRefs": {
                    "manifest": "manifest.yaml",
                    "target": "target.yaml",
                    "tests": "test.yaml",
                    "gate": "regression-gate.yaml",
                },
            },
        )
        assert client.post("/api/projects/assurance/preflight").status_code == 200
        workflow_id = client.post(
            "/api/workflows",
            json={
                "projectId": "assurance",
                "name": "Failure closure",
                "idempotencyKey": "assurance-workflow",
            },
        ).json()["workflow_id"]
        campaign = client.post(
            f"/api/workflows/{workflow_id}/campaigns",
            json={
                "idempotencyKey": "failing-campaign",
                "targets": [{"role": "baseline", "path": "target.yaml"}],
                "replayMode": "record",
            },
        ).json()
        started = client.post(
            f"/api/campaigns/{campaign['campaign_id']}/start",
            json={
                "configHash": campaign["config_hash"],
                "idempotencyKey": "start-failing-campaign",
            },
        )
        assert started.status_code == 202
        final: dict[str, object] = {}
        for _ in range(150):
            final = client.get(f"/api/campaigns/{campaign['campaign_id']}").json()
            if final["status"] in {"COMPLETED", "FAILED", "INCOMPLETE"}:
                break
            time.sleep(0.02)
        assert final["status"] == "COMPLETED"
        run_id = final["subruns"][0]["run_id"]  # type: ignore[index]

        inbox = client.get(f"/api/workflows/{workflow_id}/failures")
        assert inbox.status_code == 200
        failure = inbox.json()["failures"][0]
        assert failure["analysis_status"] == "NOT_ANALYZED"
        assert failure["failed_evaluators"] == ["schema"]
        assert "/runs/" in failure["replay_cassette_ref"]
        assert failure["replay_cassette_ref"].endswith(".jsonl")
        assert not Path(failure["replay_cassette_ref"]).is_absolute()
        filtered = client.get(
            f"/api/workflows/{workflow_id}/failures",
            params={"risk": "medium", "evaluator": "schema", "target": "baseline"},
        )
        assert len(filtered.json()["failures"]) == 1

        preview = client.get(
            f"/api/workflows/{workflow_id}/failures/{failure['failure_id']}/evidence-preview"
        )
        assert preview.status_code == 200
        evidence_hash = preview.json()["evidence_hash"]
        assert preview.json()["reusable_analyses"] == []
        assert any(
            item["evidence_id"] == "target-result" for item in preview.json()["bundle"]["items"]
        )

        queued = client.post(
            f"/api/workflows/{workflow_id}/failures/{failure['failure_id']}/analyses",
            json={"idempotencyKey": "analysis-1", "reuseExisting": False},
        )
        assert queued.status_code == 202
        step_id = queued.json()["step"]["step_id"]
        for _ in range(100):
            workflow = client.get(f"/api/workflows/{workflow_id}").json()
            step = next(item for item in workflow["steps"] if item["step_id"] == step_id)
            if step["status"] in {"COMPLETED", "FAILED_SAFE"}:
                break
            time.sleep(0.02)
        assert step["status"] == "COMPLETED"
        analysis = client.get(f"/api/workflows/{workflow_id}/analyses").json()["analyses"][0]
        assert analysis["evidence_hash"] == evidence_hash
        assert analysis["original_passed"] is False

        reused = client.post(
            f"/api/workflows/{workflow_id}/failures/{failure['failure_id']}/analyses",
            json={"idempotencyKey": "analysis-reuse", "reuseExisting": True},
        )
        assert reused.json()["reused"] is True
        assert reused.json()["report"]["analysis_id"] == analysis["analysis_id"]

        handoff = client.post(
            f"/api/workflows/{workflow_id}/handoffs",
            json={
                "analysisId": analysis["analysis_id"],
                "developerTask": "Fix the cited answer mapping and preserve other behavior.",
            },
        )
        assert handoff.status_code == 201
        assert handoff.json()["artifact"]["checksum"].startswith("sha256:")
        assert (
            client.get(f"/api/workflows/{workflow_id}").json()["status"]
            == "WAITING_EXTERNAL_REPAIR"
        )
        assert client.post(f"/api/workflows/{workflow_id}/repair-preflight").status_code == 200
        assert client.post(f"/api/workflows/{workflow_id}/resume").json()["status"] == "READY"

        compared = client.post(
            f"/api/workflows/{workflow_id}/comparisons",
            json={
                "baselineRunId": run_id,
                "candidateRunId": run_id,
                "idempotencyKey": "compare-same-run",
            },
        )
        assert compared.status_code == 200
        assert compared.json()["report"]["counts"]["persistent_failures"] == 1

        regression_gate = client.post(
            f"/api/workflows/{workflow_id}/gates",
            json={
                "kind": "regression",
                "gatePath": "regression-gate.yaml",
                "baselineRunId": run_id,
                "candidateRunId": run_id,
                "idempotencyKey": "regression-gate",
            },
        )
        assert regression_gate.status_code == 200
        assert regression_gate.json()["gate"]["result"]["exit_code"] == 0
        mutation_gate = client.post(
            f"/api/workflows/{workflow_id}/gates",
            json={
                "kind": "mutation",
                "gatePath": "mutation-gate.yaml",
                "candidateRunId": run_id,
                "idempotencyKey": "mutation-gate",
            },
        )
        assert mutation_gate.status_code == 200
        assert mutation_gate.json()["gate"]["result"]["exit_code"] == 1

        exported = client.post(
            f"/api/workflows/{workflow_id}/reports",
            json={"idempotencyKey": "report-1"},
        )
        assert exported.status_code == 200, exported.text
        body = exported.json()
        assert body["report"]["schema_version"] == "agentqa.workflow-report/v1alpha1"
        assert body["report"]["analyses"]["total"] == 1
        report_path = tmp_path / body["artifact"]["uri"]
        assert (
            body["artifact"]["checksum"]
            == "sha256:" + hashlib.sha256(report_path.read_bytes()).hexdigest()
        )

        # A failed Campaign must remain terminal, but a completed, verified
        # subrun's failure can still receive non-authoritative diagnosis.
        break_step = asyncio.run(
            workflow_store.create_step(
                workflow_id=workflow_id,
                kind="PREFLIGHT",
                idempotency_key="simulate-integrity-error",
                input_data={},
            )
        )
        asyncio.run(workflow_store.start_step(break_step.step_id))
        asyncio.run(
            workflow_store.fail_step(
                break_step.step_id,
                ErrorInfo(code="ARTIFACT_CATALOG_CHECKSUM_ERROR", message="checksum mismatch"),
            )
        )
        diagnostic = client.post(
            f"/api/workflows/{workflow_id}/failures/{failure['failure_id']}/analyses",
            json={"idempotencyKey": "diagnose-after-failure", "reuseExisting": False},
        )
        assert diagnostic.status_code == 202, diagnostic.text
        diagnostic_step_id = diagnostic.json()["step"]["step_id"]
        for _ in range(100):
            current = client.get(f"/api/workflows/{workflow_id}").json()
            diagnostic_step = next(
                item for item in current["steps"] if item["step_id"] == diagnostic_step_id
            )
            if diagnostic_step["status"] in {"COMPLETED", "FAILED_SAFE"}:
                break
            time.sleep(0.02)
        assert diagnostic_step["status"] == "COMPLETED"
        assert current["status"] == "FAILED_SAFE"
        assert current["error"]["code"] == "ARTIFACT_CATALOG_CHECKSUM_ERROR"


def test_api_imports_catalog_and_requires_explicit_code_evidence_selection(
    tmp_path: Path,
) -> None:
    reference_source = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "reference-results"
        / "t16-deepseek-20261014"
    )
    reference_target = tmp_path / "benchmark" / "reference-results" / "t16"
    reference_target.mkdir(parents=True)
    for name in (
        "generation-report.json",
        "generation-report.json.sha256",
        "mcp-transcript.json",
        "mcp-transcript.json.sha256",
        "regression-gate.json",
        "regression-gate.json.sha256",
        "plan.json",
        "plan.json.sha256",
    ):
        shutil.copy2(reference_source / name, reference_target / name)
    source_directory = tmp_path / "src"
    source_directory.mkdir()
    (source_directory / "agent.py").write_text(
        "def answer():\n    return 'safe'\n", encoding="utf-8"
    )
    (tmp_path / ".env.local").write_text("DEEPSEEK_API_KEY=secret\n", encoding="utf-8")

    store = FileResultStore(tmp_path / "store")
    app = create_app(_settings(tmp_path), store=store)
    with TestClient(app) as client:
        created = client.post(
            "/api/projects",
            json={"projectId": "catalog", "name": "Catalog project"},
        )
        assert created.status_code == 201

        imported = client.post(
            "/api/projects/catalog/artifacts/import",
            json={"sources": ["reference_results"]},
        )
        assert imported.status_code == 200
        assert imported.json()["imported"] == 4
        catalog_response = client.get("/api/projects/catalog/artifacts").json()
        catalog = catalog_response["artifacts"]
        assert catalog_response["total"] == 4
        assert (
            len(
                client.get("/api/projects/catalog/artifacts", params={"limit": 2}).json()[
                    "artifacts"
                ]
            )
            == 2
        )
        assert {item["artifact_type"] for item in catalog} == {
            "GateResult",
            "GenerationPlan",
            "GenerationReport",
            "MCPTranscript",
        }
        evidence = client.get("/api/ai/evidence").json()
        assert len(evidence["generations"]) == 1

        browse = client.get("/api/projects/catalog/workspace", params={"path": "src"}).json()
        assert browse["entries"][0]["path"] == "src/agent.py"
        selection = {
            "path": "src/agent.py",
            "startLine": 1,
            "endLine": 2,
            "reason": "Inspect the response implementation.",
        }
        preview = client.post("/api/projects/catalog/code-evidence/preview", json=selection)
        assert preview.status_code == 200
        assert preview.json()["contentHash"].startswith("sha256:")
        added = client.post("/api/projects/catalog/code-evidence", json=selection)
        assert added.status_code == 200
        boundary = client.get("/api/projects/catalog/evidence-preview").json()
        assert boundary["evidence_hash"].startswith("sha256:")
        assert boundary["code_evidence"][0]["contentHash"] == preview.json()["contentHash"]

        blocked = client.post(
            "/api/projects/catalog/code-evidence/preview",
            json={**selection, "path": ".env.local", "endLine": 1},
        )
        assert blocked.status_code == 422
        assert blocked.json()["detail"]["code"] == "CODE_EVIDENCE_PATH_BLOCKED"


def test_api_background_run_uses_workspace_relative_paths(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: Target
metadata:
  name: api-python
type: python
entrypoint: tests.fixtures.python_targets:echo
""",
        encoding="utf-8",
    )
    test = tmp_path / "test.yaml"
    test.write_text(
        """apiVersion: agentqa.dev/v1alpha1
kind: AgentTest
metadata:
  name: api-case
targetRef: api-python
scenario:
  input:
    message: hello
expect:
  output:
    requireNonEmpty: true
evaluators:
  - type: schema
  - type: llm_judge
    rubric: The answer must contain a useful response.
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """apiVersion: agentqa.dev/v1alpha2
kind: AgentManifest
metadata:
  name: api-python
runtime:
  language: python
  framework: custom
  entrypoint: tests.fixtures.python_targets:echo
purpose:
  summary: Echo a bounded API test input.
""",
        encoding="utf-8",
    )
    store = FileResultStore(tmp_path / "store")
    app = create_app(
        _settings(tmp_path),
        store=store,
        judge_provider=_PassJudgeProvider(),
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/runs",
            json={
                "testsPath": "test.yaml",
                "targetPath": "target.yaml",
                "manifestPath": "manifest.yaml",
                "judgeEnabled": True,
            },
        )
        assert created.status_code == 202
        run_id = created.json()["runId"]

        detail = None
        for _ in range(100):
            detail = client.get(f"/api/runs/{run_id}")
            body = detail.json()
            if body["state"] == "stored":
                break
            time.sleep(0.02)
        assert detail is not None
        assert detail.status_code == 200
        assert detail.json()["state"] == "stored"
        assert detail.json()["storedRun"]["run"]["status"] == "COMPLETED"
        run_record = json.loads(
            (tmp_path / "api-artifacts" / "runs" / run_id / "run.json").read_text(encoding="utf-8")
        )
        assert run_record["manifest_snapshot_ref"] == "resolved-manifest.json"
        assert run_record["manifest_status"] == "UNVERIFIED"
        results = client.get(f"/api/runs/{run_id}/results").json()["results"]
        judge = next(
            evaluation
            for evaluation in results[0]["evaluations"]
            if evaluation["evaluator_type"] == "llm_judge"
        )
        assert judge["status"] == "NEEDS_REVIEW"
        assert judge["metadata"]["pass_withheld"] is True
        assert "trace is missing" in str(judge["metadata"]["blocking_evidence_gaps"])
        assert judge["metadata"]["provider"] == "fake-deepseek"

        rejected = client.post(
            "/api/runs",
            json={"testsPath": "../outside.yaml", "targetPath": "target.yaml"},
        )
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["code"] == "RUN_REQUEST_PATH_ERROR"


def test_fastapi_otlp_endpoint_enforces_protocol_and_limits(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), store=FileResultStore(tmp_path / "store"))

    with TestClient(app) as client:
        wrong_type = client.post("/v1/traces", content=b"data")
        malformed = client.post(
            "/v1/traces",
            content=b"not protobuf",
            headers={"content-type": "application/x-protobuf"},
        )
        valid = client.post(
            "/v1/traces",
            content=ExportTraceServiceRequest().SerializeToString(),
            headers={"content-type": "application/x-protobuf"},
        )
        too_large = client.post(
            "/v1/traces",
            content=b"x" * 1025,
            headers={"content-type": "application/x-protobuf"},
        )

    assert wrong_type.status_code == 415
    assert malformed.status_code == 400
    assert valid.status_code == 200
    assert valid.headers["content-type"].startswith("application/x-protobuf")
    assert too_large.status_code == 413
