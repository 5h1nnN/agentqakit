from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from agentqa.manifest import capture_run_manifest
from agentqa.models import AgentManifest
from agentqa.specs import hash_spec, load_spec_file, normalize_spec


def _manifest(**updates: object) -> AgentManifest:
    values: dict[str, object] = {
        "apiVersion": "agentqa.dev/v1alpha2",
        "kind": "AgentManifest",
        "metadata": {"name": "test-agent", "version": "1.0.0"},
        "source": {
            "repository": "https://example.test/test-agent",
            "commit": "abc123",
            "dirty": False,
        },
        "runtime": {
            "language": "python",
            "framework": "custom",
            "entrypoint": "example.agent:run",
        },
        "model": {
            "provider": "deepseek",
            "name": "deepseek-chat",
            "parameters": {"temperature": 0.0},
        },
        "prompts": {
            "systemPromptHash": "sha256:system",
            "assetHashes": {"routing": "sha256:routing"},
        },
        "purpose": {"summary": "Complete bounded test tasks."},
        "capabilities": [{"name": "answer", "description": "Answer one bounded request."}],
        "tools": [
            {
                "name": "submit",
                "description": "Submit an external change.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"confirmed": {"type": "boolean"}},
                },
                "sideEffect": "external-write",
                "risk": "high",
                "idempotent": False,
                "requiresConfirmation": True,
            }
        ],
        "humanInteraction": {"allowed": True, "maxInterventions": 1},
        "policies": {
            "invariants": [
                {
                    "id": "confirm-submit",
                    "description": "Submission requires explicit confirmation.",
                    "severity": "high",
                }
            ]
        },
        "quality": {
            "objectives": ["task-success", "safety"],
            "budgets": {"maxTotalTokens": 2000},
        },
        "provenance": {
            "defaultLevel": "declared",
            "declaredBy": "test author",
            "reviewed": True,
        },
    }
    values.update(updates)
    return AgentManifest.model_validate(values)


def test_v1alpha1_manifest_migrates_without_inventing_semantics(tmp_path: Path) -> None:
    source = tmp_path / "manifest.yaml"
    source.write_text(
        dedent(
            """
            apiVersion: agentqa.dev/v1alpha1
            kind: AgentManifest
            metadata:
              name: legacy-agent
            runtime:
              language: python
            tools:
              - name: lookup
                schemaHash: sha256:legacy
            """
        ).lstrip(),
        encoding="utf-8",
    )

    loaded = load_spec_file(source)[0]

    assert isinstance(loaded.spec, AgentManifest)
    normalized = normalize_spec(loaded.spec)
    assert normalized["apiVersion"] == "agentqa.dev/v1alpha2"
    assert normalized["tools"][0]["sideEffect"] == "none"
    assert normalized["provenance"]["declaredBy"] == "v1alpha1 migration"
    assert normalized["purpose"] is None


def test_manifest_rejects_ambiguous_side_effects_and_duplicate_capabilities() -> None:
    with pytest.raises(ValidationError, match="require risk and idempotent"):
        _manifest(
            tools=[
                {
                    "name": "write",
                    "sideEffect": "external-write",
                }
            ]
        )

    with pytest.raises(ValidationError, match="capability names must be unique"):
        _manifest(
            capabilities=[
                {"name": "answer", "description": "First."},
                {"name": "answer", "description": "Second."},
            ]
        )


def test_manifest_requires_human_policy_for_confirmation_tools() -> None:
    with pytest.raises(ValidationError, match=r"humanInteraction\.allowed"):
        _manifest(humanInteraction={"allowed": False})


def test_manifest_rejects_unknown_fields_and_hashes_canonically() -> None:
    manifest = _manifest()

    with pytest.raises(ValidationError, match="extra_forbidden"):
        _manifest(unknownField=True)

    reordered = AgentManifest.model_validate(manifest.model_dump(by_alias=True, mode="json"))
    assert hash_spec(manifest) == hash_spec(reordered)


def test_runtime_snapshot_reports_verified_and_stale_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()

    def verified_git_value(_workspace: Path, *arguments: str) -> str | None:
        return "abc123" if arguments[:2] == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr("agentqa.manifest.runtime._git_value", verified_git_value)
    verified = capture_run_manifest(manifest)

    assert verified.status == "VERIFIED"
    assert verified.source.actual_commit == "abc123"
    assert verified.runtime_contract.system_prompt_hash == "sha256:system"
    assert verified.runtime_contract.prompt_asset_hashes == {"routing": "sha256:routing"}
    assert verified.runtime_contract.tool_schemas["submit"].computed_hash is not None
    assert verified.runtime_contract.model is not None
    assert verified.runtime_contract.model.parameters == {"temperature": 0.0}

    def stale_git_value(_workspace: Path, *arguments: str) -> str | None:
        return "different" if arguments[:2] == ("rev-parse", "HEAD") else "modified"

    monkeypatch.setattr("agentqa.manifest.runtime._git_value", stale_git_value)
    stale = capture_run_manifest(manifest)

    assert stale.status == "STALE_MANIFEST"
    assert len(stale.warnings) == 2
