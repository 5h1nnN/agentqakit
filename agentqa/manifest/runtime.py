"""Capture a bounded, read-only Git snapshot for manifest drift detection."""

import subprocess
from pathlib import Path
from typing import Literal

from agentqa.conventions import utc_now
from agentqa.models.manifest import (
    ModelRuntimeSnapshot,
    RunManifestSnapshot,
    RuntimeContractSnapshot,
    SourceVerification,
    ToolSchemaSnapshot,
)
from agentqa.models.specs import AgentManifest
from agentqa.specs.hashing import canonical_json_bytes, hash_spec, sha256_digest

_GIT_TIMEOUT_SECONDS = 2.0


def _git_value(workspace: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _runtime_contract(manifest: AgentManifest) -> RuntimeContractSnapshot:
    system_prompt_hash = None if manifest.prompts is None else manifest.prompts.system_prompt_hash
    prompt_asset_hashes = {} if manifest.prompts is None else manifest.prompts.asset_hashes
    tool_schemas: dict[str, ToolSchemaSnapshot] = {}
    for tool in manifest.tools:
        computed_hash = None
        if tool.input_schema is not None or tool.output_schema is not None:
            computed_hash = sha256_digest(
                canonical_json_bytes(
                    {
                        "inputSchema": tool.input_schema,
                        "outputSchema": tool.output_schema,
                    }
                )
            )
        tool_schemas[tool.name] = ToolSchemaSnapshot(
            declared_hash=tool.schema_hash,
            computed_hash=computed_hash,
        )
    model = (
        None
        if manifest.model is None
        else ModelRuntimeSnapshot(
            provider=manifest.model.provider,
            name=manifest.model.name,
            parameters=manifest.model.parameters,
        )
    )
    return RuntimeContractSnapshot(
        system_prompt_hash=system_prompt_hash,
        prompt_asset_hashes=prompt_asset_hashes,
        tool_schemas=tool_schemas,
        model=model,
    )


def capture_run_manifest(
    manifest: AgentManifest,
    *,
    workspace: str | Path | None = None,
) -> RunManifestSnapshot:
    """Compare declared source identity with the local Git checkout when available."""
    root = Path.cwd() if workspace is None else Path(workspace)
    actual_commit = _git_value(root, "rev-parse", "HEAD")
    status_output = _git_value(root, "status", "--porcelain", "--untracked-files=normal")
    actual_dirty = None if status_output is None else bool(status_output)
    declared_commit = None if manifest.source is None else manifest.source.commit
    declared_dirty = None if manifest.source is None else manifest.source.dirty
    warnings: list[str] = []

    status: Literal["VERIFIED", "STALE_MANIFEST", "UNVERIFIED"]
    if actual_commit is None:
        status = "UNVERIFIED"
        warnings.append("local Git source identity could not be verified")
    else:
        stale = False
        if declared_commit is None:
            warnings.append("manifest does not declare a source commit")
            status = "UNVERIFIED"
        else:
            if declared_commit != actual_commit:
                stale = True
                warnings.append("declared source commit differs from the local checkout")
            if (
                declared_dirty is not None
                and actual_dirty is not None
                and declared_dirty != actual_dirty
            ):
                stale = True
                warnings.append("declared dirty state differs from the local checkout")
            status = "STALE_MANIFEST" if stale else "VERIFIED"

    return RunManifestSnapshot(
        captured_at=utc_now(),
        declared_manifest_hash=hash_spec(manifest),
        status=status,
        source=SourceVerification(
            declared_commit=declared_commit,
            actual_commit=actual_commit,
            declared_dirty=declared_dirty,
            actual_dirty=actual_dirty,
        ),
        runtime_contract=_runtime_contract(manifest),
        warnings=warnings,
    )
