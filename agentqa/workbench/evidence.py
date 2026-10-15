"""Deterministic preview of the evidence boundary for future AI operations."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from agentqa.evaluators import JudgeProviderLike
from agentqa.models import AgentManifest
from agentqa.specs import SpecLoadError, hash_spec, load_specs
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest
from agentqa.workbench.catalog import ArtifactCatalogService
from agentqa.workbench.models import EvidenceBoundaryPreview, WorkbenchProject


class EvidenceBoundaryService:
    def __init__(
        self,
        workspace_root: str | Path,
        catalog: ArtifactCatalogService,
        judge_provider: JudgeProviderLike | None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.catalog = catalog
        self.judge_provider = judge_provider

    def _manifest(self, project: WorkbenchProject) -> AgentManifest | None:
        reference = project.default_spec_refs.get("manifest")
        if reference is None:
            return None
        project_root = (
            self.workspace_root / Path(*PurePosixPath(project.workspace_path).parts)
        ).resolve()
        path = (project_root / Path(*PurePosixPath(reference).parts)).resolve()
        try:
            path.relative_to(project_root)
            loaded = load_specs(path)
        except (ValueError, SpecLoadError):
            return None
        return next((item.spec for item in loaded if isinstance(item.spec, AgentManifest)), None)

    async def preview(self, project: WorkbenchProject) -> EvidenceBoundaryPreview:
        manifest = self._manifest(project)
        code_evidence = await self.catalog.list_code_evidence(project.project_id)
        manifest_sections: list[str] = []
        manifest_hash = None
        if manifest is not None:
            manifest_hash = hash_spec(manifest)
            value = manifest.model_dump(mode="json", by_alias=True, exclude_none=True)
            manifest_sections = sorted(
                key for key in value if key not in {"apiVersion", "kind", "metadata"}
            )
        evidence_types = ["manifest-summary"] if manifest is not None else []
        if code_evidence:
            evidence_types.append("reviewed-code-evidence")
        material = {
            "project_id": project.project_id,
            "manifest_hash": manifest_hash,
            "manifest_sections": manifest_sections,
            "code_evidence": [
                {
                    "path": item.path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "content_hash": item.content_hash,
                }
                for item in code_evidence
            ],
            "evidence_types": evidence_types,
            "provider": None if self.judge_provider is None else self.judge_provider.name,
            "model": None if self.judge_provider is None else self.judge_provider.model,
            "max_input_tokens": 12_000,
            "max_output_tokens": 4_096,
            "timeout_seconds": 60.0,
        }
        return EvidenceBoundaryPreview(
            project_id=project.project_id,
            manifest_sections=manifest_sections,
            code_evidence=code_evidence,
            evidence_types=evidence_types,
            excluded=[
                "API keys and credential-like fields",
                "unselected workspace files",
                "complete raw traces",
                "trace summaries and evaluations until a specific trial is selected",
                "unbounded tool or model output",
            ],
            provider=None if self.judge_provider is None else self.judge_provider.name,
            model=None if self.judge_provider is None else self.judge_provider.model,
            max_input_tokens=12_000,
            max_output_tokens=4_096,
            timeout_seconds=60.0,
            evidence_hash=sha256_digest(canonical_json_bytes(material)),
        )


__all__ = ["EvidenceBoundaryService"]
