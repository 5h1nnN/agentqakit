"""Version-dispatch entry point for future specification migrations."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from agentqa.models import API_VERSION

Migration = Callable[[dict[str, Any]], dict[str, Any]]

# Future versions register explicit one-step migrations here. The first public
# alpha format is already canonical, so no migration is currently necessary.
MIGRATIONS: dict[str, Migration] = {}


class UnsupportedSpecVersionError(ValueError):
    """Raised when no safe migration path exists for a specification."""


def migrate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return a migrated copy of *document* at the current API version."""
    migrated = deepcopy(document)
    version = migrated.get("apiVersion")
    visited: set[str] = set()
    while version != API_VERSION:
        if not isinstance(version, str) or version in visited or version not in MIGRATIONS:
            raise UnsupportedSpecVersionError(f"unsupported apiVersion: {version!r}")
        visited.add(version)
        migrated = MIGRATIONS[version](migrated)
        version = migrated.get("apiVersion")
    return migrated
