import asyncio
from pathlib import Path

import pytest

from agentqa.mcp import AgentQAMCPApplication
from agentqa.storage import FileResultStore
from agentqa.workbench import (
    ArtifactCatalogService,
    CodeEvidenceError,
    CodeEvidenceService,
    ProjectPreflightService,
    SQLiteWorkflowStore,
    WorkbenchProject,
)
from tests.fixtures.stored_runs import create_storable_run


def test_catalog_imports_verified_run_and_registers_it_in_result_store(tmp_path: Path) -> None:
    async def exercise() -> None:
        artifacts = tmp_path / "artifacts"
        run, _ = create_storable_run(artifacts, "catalog-run", passed=True)
        result_store = FileResultStore(tmp_path / "results")
        await result_store.initialize()
        catalog = ArtifactCatalogService(
            tmp_path / "catalog.sqlite3",
            workspace_root=tmp_path,
            result_store=result_store,
            artifact_directory=artifacts,
        )
        await catalog.initialize()

        imported = await catalog.discover("campus", ["artifacts"])
        assert imported.runs_registered == 1
        assert imported.imported >= 4
        assert (await result_store.get_run(run.run_id)) is not None
        entries = await catalog.list_artifacts("campus")
        assert any(entry.artifact_type == "RunLifecycle" for entry in entries)
        assert all(entry.availability == "AVAILABLE" for entry in entries)

        repeated = await catalog.discover("campus", ["artifacts"])
        assert repeated.runs_registered == 0
        assert repeated.unchanged == imported.imported

        lifecycle = next(entry for entry in entries if entry.artifact_type == "RunLifecycle")
        (tmp_path / lifecycle.path).write_text("{}", encoding="utf-8")
        refreshed = await catalog.list_artifacts("campus")
        changed = next(entry for entry in refreshed if entry.artifact_id == lifecycle.artifact_id)
        assert changed.availability == "INVALID"

    asyncio.run(exercise())


def test_catalog_imports_mcp_artifact_with_the_authoritative_checksum(tmp_path: Path) -> None:
    async def exercise() -> None:
        artifacts = tmp_path / "artifacts"
        create_storable_run(artifacts, "baseline", passed=False)
        create_storable_run(artifacts, "candidate", passed=True)
        application = AgentQAMCPApplication(tmp_path)
        result = application.compare_runs("baseline", "candidate")
        report_path = tmp_path / result["report_path"]
        checksum = report_path.with_name(f"{report_path.name}.sha256").read_text().strip()

        result_store = FileResultStore(tmp_path / "results")
        await result_store.initialize()
        catalog = ArtifactCatalogService(
            tmp_path / "catalog.sqlite3",
            workspace_root=tmp_path,
            result_store=result_store,
            artifact_directory=artifacts,
        )
        await catalog.initialize()
        imported = await catalog.discover("campus", ["mcp"])
        entries = await catalog.list_artifacts("campus", "ComparisonReport")

        assert imported.imported == 1
        assert len(entries) == 1
        assert entries[0].source == "MCP"
        assert entries[0].checksum == checksum

    asyncio.run(exercise())


def test_code_evidence_is_explicit_bounded_redacted_and_workspace_confined(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "agent"
    project_root.mkdir()
    source = project_root / "agent.py"
    source.write_text(
        "def answer():\n    token = 'safe-placeholder'\n    return 'ok'  # api_key=secret-value\n",
        encoding="utf-8",
    )
    (project_root / ".env.local").write_text("DEEPSEEK_API_KEY=secret", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    (project_root / "linked.py").symlink_to(outside)
    store = SQLiteWorkflowStore(tmp_path / "projects.sqlite3")

    async def create_project() -> WorkbenchProject:
        await store.initialize()
        return await store.create_project("campus", "Campus", workspace_path="agent")

    project = asyncio.run(create_project())
    service = CodeEvidenceService(tmp_path)
    entries = service.browse(project)
    assert [entry.path for entry in entries] == ["agent.py"]

    evidence = service.preview(
        project,
        path="agent.py",
        start_line=1,
        end_line=3,
        reason="Inspect the bounded response implementation.",
    )
    assert evidence.path == "agent/agent.py"
    assert "secret-value" not in evidence.content
    assert "api_key=***" in evidence.content

    with pytest.raises(CodeEvidenceError) as environment:
        service.preview(
            project,
            path=".env.local",
            start_line=1,
            end_line=1,
            reason="must fail",
        )
    assert environment.value.code == "CODE_EVIDENCE_PATH_BLOCKED"
    with pytest.raises(CodeEvidenceError):
        service.preview(
            project,
            path="../outside.py",
            start_line=1,
            end_line=1,
            reason="must fail",
        )
    with pytest.raises(CodeEvidenceError):
        service.preview(
            project,
            path="linked.py",
            start_line=1,
            end_line=1,
            reason="must fail",
        )


def test_campus_project_preflight_separates_blocks_and_reviews(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]

    async def exercise() -> None:
        store = SQLiteWorkflowStore(tmp_path / "preflight.sqlite3")
        await store.initialize()
        project = await store.create_project(
            "campus",
            "Campus Ops",
            workspace_path="examples/showcase/campus_ops_agent",
            default_spec_refs={
                "manifest": "agentqa/manifest-v2.yaml",
                "target": "agentqa/targets/fixed.yaml",
                "tests": "agentqa/golden-tests/campus-ops.yaml",
                "mutations": "agentqa/mutations.yaml",
                "gate": "agentqa/quality-gate.yaml",
                "generation_goal": "agentqa/generation-goal.yaml",
            },
        )
        report = await ProjectPreflightService(repository, store, None).run(project)
        assert report.executable is True
        assert report.requires_approval is True
        assert not [check for check in report.checks if check.status == "BLOCK"]
        assert report.manifest_snapshot is not None
        persisted = await store.get_project("campus")
        assert persisted is not None
        assert persisted.last_manifest_snapshot == report.manifest_snapshot

    asyncio.run(exercise())
