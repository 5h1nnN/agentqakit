"""Read-only project readiness checks for the local assurance workbench."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from agentqa.conventions import utc_now
from agentqa.evaluators import JudgeProviderLike
from agentqa.manifest import capture_run_manifest
from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    GenerationGoal,
    MutationSpec,
    QualityGateSpec,
)
from agentqa.models.specs import HttpTargetSpec, LangGraphTargetSpec, PythonTargetSpec
from agentqa.specs import SpecLoadError, load_specs
from agentqa.workbench.models import (
    PreflightCheck,
    PreflightReport,
    WorkbenchProject,
)
from agentqa.workbench.store import WorkflowStore

_REQUIRED_REFS = ("manifest", "target", "tests")
_EXPECTED_TYPES: dict[str, tuple[type[Any], ...]] = {
    "manifest": (AgentManifest,),
    "target": (HttpTargetSpec, PythonTargetSpec, LangGraphTargetSpec),
    "baseline_target": (HttpTargetSpec, PythonTargetSpec, LangGraphTargetSpec),
    "candidate_target": (HttpTargetSpec, PythonTargetSpec, LangGraphTargetSpec),
    "defect_target": (HttpTargetSpec, PythonTargetSpec, LangGraphTargetSpec),
    "fixed_target": (HttpTargetSpec, PythonTargetSpec, LangGraphTargetSpec),
    "tests": (AgentTestSpec,),
    "mutations": (MutationSpec,),
    "gate": (QualityGateSpec,),
    "generation_goal": (GenerationGoal,),
}


class ProjectPreflightService:
    def __init__(
        self,
        workspace_root: str | Path,
        store: WorkflowStore,
        judge_provider: JudgeProviderLike | None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.store = store
        self.judge_provider = judge_provider

    def _project_root(self, project: WorkbenchProject) -> Path:
        relative = PurePosixPath(project.workspace_path)
        root = (self.workspace_root / Path(*relative.parts)).resolve()
        try:
            root.relative_to(self.workspace_root)
        except ValueError:
            return self.workspace_root / "__invalid_project_workspace__"
        return root

    @staticmethod
    def _entrypoint_exists(project_root: Path, entrypoint: str) -> bool:
        module_name = entrypoint.split(":", 1)[0]
        module_path = Path(*module_name.split("."))
        candidates = (
            project_root / f"{module_path}.py",
            project_root / module_path / "__init__.py",
            project_root / "src" / f"{module_path}.py",
            project_root / "src" / module_path / "__init__.py",
        )
        return any(candidate.is_file() for candidate in candidates)

    async def run(self, project: WorkbenchProject) -> PreflightReport:
        checks: list[PreflightCheck] = []
        project_root = self._project_root(project)
        if not project_root.is_dir():
            checks.append(
                PreflightCheck(
                    check_id="workspace-path",
                    category="path",
                    status="BLOCK",
                    message="Project workspace does not exist inside the fixed server workspace.",
                    path=project.workspace_path,
                )
            )
            return PreflightReport(
                project_id=project.project_id,
                checked_at=utc_now(),
                executable=False,
                requires_approval=False,
                checks=checks,
            )
        checks.append(
            PreflightCheck(
                check_id="workspace-path",
                category="path",
                status="PASS",
                message="Project workspace is contained in the fixed server workspace.",
                path=project.workspace_path,
            )
        )

        for required in _REQUIRED_REFS:
            if required not in project.default_spec_refs:
                checks.append(
                    PreflightCheck(
                        check_id=f"required-{required}",
                        category="path",
                        status="BLOCK",
                        message=f"Required project reference '{required}' is not configured.",
                    )
                )

        manifest: AgentManifest | None = None
        entrypoints: list[tuple[str, str]] = []
        for name, raw_path in sorted(project.default_spec_refs.items()):
            expected = _EXPECTED_TYPES.get(name)
            if expected is None:
                checks.append(
                    PreflightCheck(
                        check_id=f"spec-{name}",
                        category="schema",
                        status="REVIEW",
                        message=f"Reference '{name}' has no predefined schema role.",
                        path=raw_path,
                    )
                )
                continue
            relative = PurePosixPath(raw_path)
            candidate = (project_root / Path(*relative.parts)).resolve()
            try:
                candidate.relative_to(project_root)
            except ValueError:
                checks.append(
                    PreflightCheck(
                        check_id=f"path-{name}",
                        category="path",
                        status="BLOCK",
                        message=f"Reference '{name}' escapes the project workspace.",
                        path=raw_path,
                    )
                )
                continue
            if not candidate.exists():
                checks.append(
                    PreflightCheck(
                        check_id=f"path-{name}",
                        category="path",
                        status="BLOCK",
                        message=f"Reference '{name}' does not exist.",
                        path=raw_path,
                    )
                )
                continue
            try:
                loaded = load_specs(candidate)
            except SpecLoadError as exc:
                checks.append(
                    PreflightCheck(
                        check_id=f"schema-{name}",
                        category="schema",
                        status="BLOCK",
                        message=f"Reference '{name}' failed schema validation: {exc.message[:700]}",
                        path=raw_path,
                    )
                )
                continue
            invalid = [item for item in loaded if not isinstance(item.spec, expected)]
            if not loaded or invalid:
                checks.append(
                    PreflightCheck(
                        check_id=f"schema-{name}",
                        category="schema",
                        status="BLOCK",
                        message=f"Reference '{name}' contains an unexpected specification kind.",
                        path=raw_path,
                    )
                )
                continue
            checks.append(
                PreflightCheck(
                    check_id=f"schema-{name}",
                    category="schema",
                    status="PASS",
                    message=f"Reference '{name}' passed strict schema validation.",
                    path=raw_path,
                )
            )
            for item in loaded:
                if isinstance(item.spec, AgentManifest):
                    manifest = item.spec
                    if manifest.runtime.entrypoint is not None:
                        entrypoints.append(("manifest", manifest.runtime.entrypoint))
                elif isinstance(item.spec, PythonTargetSpec | LangGraphTargetSpec):
                    entrypoints.append((name, item.spec.entrypoint))

        for index, (source, entrypoint) in enumerate(entrypoints, start=1):
            exists = self._entrypoint_exists(project_root, entrypoint)
            checks.append(
                PreflightCheck(
                    check_id=f"entrypoint-{index}",
                    category="entrypoint",
                    status="PASS" if exists else "BLOCK",
                    message=(
                        f"Entrypoint declared by {source} resolves to local source."
                        if exists
                        else f"Entrypoint declared by {source} has no matching local module."
                    ),
                    path=entrypoint,
                )
            )

        dependency_files = [
            path
            for path in (project_root / "pyproject.toml", project_root / "requirements.txt")
            if path.is_file()
        ]
        checks.append(
            PreflightCheck(
                check_id="dependencies",
                category="dependency",
                status="PASS" if dependency_files else "REVIEW",
                message=(
                    "A local dependency declaration is available."
                    if dependency_files
                    else (
                        "No supported dependency declaration was found; "
                        "runtime availability needs review."
                    )
                ),
                path=(
                    dependency_files[0].relative_to(project_root).as_posix()
                    if dependency_files
                    else None
                ),
            )
        )
        checks.append(
            PreflightCheck(
                check_id="provider",
                category="provider",
                status="PASS" if self.judge_provider is not None else "REVIEW",
                message=(
                    f"Structured LLM provider '{self.judge_provider.name}' is configured."
                    if self.judge_provider is not None
                    else (
                        "No structured LLM provider is configured; "
                        "deterministic workflows remain usable."
                    )
                ),
            )
        )

        snapshot = None
        if manifest is not None:
            snapshot = capture_run_manifest(manifest, workspace=project_root)
            await self.store.update_project_observation(project.project_id, snapshot)
            checks.append(
                PreflightCheck(
                    check_id="source-identity",
                    category="source",
                    status="PASS" if snapshot.status == "VERIFIED" else "REVIEW",
                    message=(
                        "Manifest source identity matches the locally observed Git state."
                        if snapshot.status == "VERIFIED"
                        else (
                            "Manifest source identity is unverified or differs "
                            "from local Git state."
                        )
                    ),
                )
            )
            risky_tools = [
                tool.name
                for tool in manifest.tools
                if tool.risk in {"high", "critical"} or tool.requires_confirmation
            ]
            side_effects = [
                tool.name
                for tool in manifest.tools
                if tool.side_effect in {"external-write", "irreversible"}
            ]
            checks.append(
                PreflightCheck(
                    check_id="tool-risk",
                    category="tool_risk",
                    status="REVIEW" if risky_tools else "PASS",
                    message=(
                        f"High-risk or confirmation tools require review: {', '.join(risky_tools)}."
                        if risky_tools
                        else "No high-risk or confirmation-required tools were declared."
                    ),
                )
            )
            checks.append(
                PreflightCheck(
                    check_id="side-effects",
                    category="side_effect",
                    status="REVIEW" if side_effects else "PASS",
                    message=(
                        "External or irreversible side effects require approval: "
                        f"{', '.join(side_effects)}."
                        if side_effects
                        else "No external or irreversible tool side effects were declared."
                    ),
                )
            )

        executable = not any(check.status == "BLOCK" for check in checks)
        requires_approval = any(check.status == "REVIEW" for check in checks)
        return PreflightReport(
            project_id=project.project_id,
            checked_at=utc_now(),
            executable=executable,
            requires_approval=requires_approval,
            checks=checks,
            manifest_snapshot=snapshot,
        )


__all__ = ["ProjectPreflightService"]
