"""Conservative project setup discovery for the local Web workbench."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from agentqa.models import (
    AgentManifest,
    AgentTestSpec,
    GenerationGoal,
    MutationSpec,
    QualityGateSpec,
)
from agentqa.models.specs import HttpTargetSpec, LangGraphTargetSpec, PythonTargetSpec
from agentqa.specs import SpecLoadError, load_specs

_SPEC_SUFFIXES = {".json", ".yaml", ".yml"}
_EXCLUDED_DIRECTORIES = {
    ".git",
    ".agentqa",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
_TARGET_TYPES = (HttpTargetSpec, PythonTargetSpec, LangGraphTargetSpec)


def _project_root(workspace_root: Path, workspace_path: str) -> Path:
    relative = PurePosixPath(workspace_path)
    root = (workspace_root / Path(*relative.parts)).resolve()
    try:
        root.relative_to(workspace_root)
    except ValueError:
        return workspace_root / "__invalid_project_workspace__"
    return root


def _rank(path: str, role: str) -> tuple[int, int, str]:
    name = Path(path).name.lower()
    score = 0
    preferred: dict[str, tuple[str, ...]] = {
        "manifest": ("manifest.yaml", "manifest.yml", "manifest-v2.yaml"),
        "target": ("target.yaml", "fixed.yaml", "candidate.yaml"),
        "tests": ("golden-tests.yaml", "tests.yaml", "campus-ops.yaml"),
        "mutations": ("mutations.yaml",),
        "gate": ("regression-gate.yaml", "quality-gate.yaml", "gate.yaml"),
        "generation_goal": ("generation-goal.yaml",),
    }
    if name in preferred.get(role, ()):
        score += 100 - preferred[role].index(name)
    if role.replace("_", "-") in name:
        score += 30
    if "minimal" in name or "comparison" in name or "code" in name:
        score -= 10
    if role == "gate" and "mutation" in name:
        score -= 25
    return (-score, len(Path(path).parts), path)


def discover_spec_refs(
    workspace_root: str | Path,
    workspace_path: str,
    *,
    max_files: int = 500,
) -> dict[str, str]:
    """Find well-typed AgentQA specs without requiring path-by-path setup."""

    fixed_root = Path(workspace_root).resolve()
    root = _project_root(fixed_root, workspace_path)
    if not root.is_dir():
        return {}

    candidates: list[Path] = []
    for path in root.rglob("*"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part.lower() in _EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in _SPEC_SUFFIXES:
            candidates.append(path)
            if len(candidates) >= max_files:
                break

    by_role: dict[str, list[str]] = {}
    target_paths: list[str] = []
    for path in sorted(candidates):
        try:
            loaded = load_specs(path)
        except (OSError, SpecLoadError):
            continue
        relative_path = path.relative_to(root).as_posix()
        if any(isinstance(item.spec, AgentManifest) for item in loaded):
            by_role.setdefault("manifest", []).append(relative_path)
        if any(isinstance(item.spec, AgentTestSpec) for item in loaded):
            by_role.setdefault("tests", []).append(relative_path)
        if any(isinstance(item.spec, MutationSpec) for item in loaded):
            by_role.setdefault("mutations", []).append(relative_path)
        if any(isinstance(item.spec, QualityGateSpec) for item in loaded):
            by_role.setdefault("gate", []).append(relative_path)
        if any(isinstance(item.spec, GenerationGoal) for item in loaded):
            by_role.setdefault("generation_goal", []).append(relative_path)
        if any(isinstance(item.spec, _TARGET_TYPES) for item in loaded):
            target_paths.append(relative_path)

    discovered = {
        role: sorted(paths, key=lambda value: _rank(value, role))[0]
        for role, paths in by_role.items()
        if paths
    }
    if target_paths:
        for value in target_paths:
            name = Path(value).name.lower()
            for role, markers in {
                "baseline_target": ("baseline",),
                "candidate_target": ("candidate",),
                "defect_target": ("bug", "defect", "broken"),
                "fixed_target": ("fixed",),
            }.items():
                if any(marker in name for marker in markers):
                    discovered.setdefault(role, value)
        preferred_target = (
            discovered.get("fixed_target")
            or discovered.get("candidate_target")
            or sorted(target_paths, key=lambda value: _rank(value, "target"))[0]
        )
        discovered["target"] = preferred_target
    return discovered


def discovery_summary(refs: dict[str, str]) -> dict[str, Any]:
    """Return a stable, UI-friendly explanation of discovered configuration."""

    required = ("manifest", "target", "tests")
    return {
        "refs": refs,
        "ready": all(item in refs for item in required),
        "missing": [item for item in required if item not in refs],
    }


__all__ = ["discover_spec_refs", "discovery_summary"]
