from pathlib import Path

from agentqa.conventions import utc_now
from agentqa.workbench.code_evidence import CodeEvidenceService
from agentqa.workbench.discovery import discover_spec_refs
from agentqa.workbench.models import WorkbenchProject


def test_discovers_showcase_configuration_from_project_directory() -> None:
    root = Path(__file__).resolve().parents[2]

    refs = discover_spec_refs(root, "examples/showcase/campus_ops_agent")

    assert refs["manifest"] == "agentqa/manifest-v2.yaml"
    assert refs["target"] == "agentqa/targets/fixed.yaml"
    assert refs["tests"] == "agentqa/golden-tests/campus-ops.yaml"
    assert refs["mutations"] == "agentqa/mutations.yaml"
    assert refs["gate"] == "agentqa/regression-gate.yaml"


def test_builds_bounded_repository_context_without_secret_files(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "agent_service.py").write_text(
        "class AgentService:\n    def answer(self, prompt: str) -> str:\n        return prompt\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text("DEEPSEEK_API_KEY=secret", encoding="utf-8")
    now = utc_now()
    project = WorkbenchProject(
        project_id="demo",
        name="Demo",
        workspace_path=".",
        created_at=now,
        updated_at=now,
    )

    evidence = CodeEvidenceService(tmp_path).discover(project)

    assert len(evidence) == 1
    assert evidence[0].path == "src/agent_service.py"
    assert "secret" not in evidence[0].content
    assert evidence[0].end_line == 3
