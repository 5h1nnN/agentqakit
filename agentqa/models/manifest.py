"""Runtime verification models for one declared agent manifest."""

from typing import Literal

from pydantic import ConfigDict, Field, JsonValue

from agentqa.models.base import NonEmptyString, ResourceName, StrictModel, UtcDateTime

RUN_MANIFEST_SCHEMA_VERSION: Literal["agentqa.run-manifest/v1alpha1"] = (
    "agentqa.run-manifest/v1alpha1"
)


class SourceVerification(StrictModel):
    """Declared and locally observed Git identity for an evaluated agent."""

    declared_commit: NonEmptyString | None = None
    actual_commit: NonEmptyString | None = None
    declared_dirty: bool | None = None
    actual_dirty: bool | None = None


class ToolSchemaSnapshot(StrictModel):
    """Declared and canonical hashes for one tool contract."""

    declared_hash: NonEmptyString | None = None
    computed_hash: NonEmptyString | None = None


class ModelRuntimeSnapshot(StrictModel):
    """Non-secret model identity and parameters resolved for the run."""

    provider: NonEmptyString
    name: NonEmptyString
    parameters: dict[str, JsonValue] = Field(default_factory=dict, max_length=100)


class RuntimeContractSnapshot(StrictModel):
    """Prompt, tool, and model identities frozen at run start."""

    system_prompt_hash: NonEmptyString | None = None
    prompt_asset_hashes: dict[ResourceName, NonEmptyString] = Field(
        default_factory=dict,
        max_length=200,
    )
    tool_schemas: dict[ResourceName, ToolSchemaSnapshot] = Field(
        default_factory=dict,
        max_length=500,
    )
    model: ModelRuntimeSnapshot | None = None


class RunManifestSnapshot(StrictModel):
    """Immutable verification snapshot captured when a test run starts."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["agentqa.run-manifest/v1alpha1"] = RUN_MANIFEST_SCHEMA_VERSION
    captured_at: UtcDateTime
    declared_manifest_hash: NonEmptyString
    status: Literal["VERIFIED", "STALE_MANIFEST", "UNVERIFIED"]
    source: SourceVerification
    runtime_contract: RuntimeContractSnapshot = Field(default_factory=RuntimeContractSnapshot)
    warnings: list[NonEmptyString] = Field(default_factory=list, max_length=20)
