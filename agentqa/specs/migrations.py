"""Kind-aware, explicit one-step migrations for public specifications."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from agentqa.models import AGENT_MANIFEST_API_VERSION, API_VERSION

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _migrate_agent_manifest_v1alpha1(document: dict[str, Any]) -> dict[str, Any]:
    """Upgrade the identity-only manifest without inventing semantic claims."""
    migrated = deepcopy(document)
    migrated["apiVersion"] = AGENT_MANIFEST_API_VERSION
    migrated.setdefault(
        "provenance",
        {
            "defaultLevel": "declared",
            "declaredBy": "v1alpha1 migration",
            "reviewed": False,
        },
    )
    return migrated


MIGRATIONS: dict[tuple[str, str], Migration] = {
    ("AgentManifest", API_VERSION): _migrate_agent_manifest_v1alpha1,
}

CURRENT_VERSION_BY_KIND: dict[str, str] = {
    "AgentManifest": AGENT_MANIFEST_API_VERSION,
}


class UnsupportedSpecVersionError(ValueError):
    """Raised when no safe migration path exists for a specification."""


def migrate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return a migrated copy of *document* at the current API version."""
    migrated = deepcopy(document)
    kind = migrated.get("kind")
    target_version = (
        CURRENT_VERSION_BY_KIND.get(kind, API_VERSION) if isinstance(kind, str) else API_VERSION
    )
    version = migrated.get("apiVersion")
    visited: set[str] = set()
    while version != target_version:
        key = (kind, version) if isinstance(kind, str) and isinstance(version, str) else None
        if (
            not isinstance(version, str)
            or version in visited
            or key is None
            or key not in MIGRATIONS
        ):
            raise UnsupportedSpecVersionError(f"unsupported apiVersion: {version!r}")
        visited.add(version)
        migrated = MIGRATIONS[key](migrated)
        version = migrated.get("apiVersion")
    return migrated
