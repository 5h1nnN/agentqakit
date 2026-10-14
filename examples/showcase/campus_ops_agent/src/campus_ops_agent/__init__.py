"""CampusOps Agent public API.

The core service deliberately does not import LangGraph. Applications that need the
adapter import factories from :mod:`campus_ops_agent.graph` explicitly.
"""

from campus_ops_agent.service import CampusOpsService, create_service, run
from campus_ops_agent.variants import Variant, VariantConfig

__all__ = [
    "CampusOpsService",
    "Variant",
    "VariantConfig",
    "create_service",
    "run",
]
