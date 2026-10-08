"""Shared AgentQAKit exceptions."""

from typing import Any


class AgentQAError(Exception):
    """Base class for expected, user-actionable AgentQAKit failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "AGENTQA_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without a traceback."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }
