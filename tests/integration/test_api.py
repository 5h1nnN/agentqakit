import asyncio
import json
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
from agentqa.storage import FileResultStore
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
