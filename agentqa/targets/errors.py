"""Stable errors and error codes produced by target adapters."""

from enum import StrEnum

from agentqa.errors import AgentQAError


class TargetErrorCode(StrEnum):
    """Failure categories shared by target adapters."""

    CONFIGURATION_ERROR = "TARGET_CONFIGURATION_ERROR"
    HTTP_SAFETY_ERROR = "HTTP_SAFETY_ERROR"
    HTTP_TIMEOUT = "HTTP_TIMEOUT"
    HTTP_CONNECTION_ERROR = "HTTP_CONNECTION_ERROR"
    HTTP_STATUS_ERROR = "HTTP_STATUS_ERROR"
    HTTP_INVALID_RESPONSE = "HTTP_INVALID_RESPONSE"
    HTTP_RESPONSE_TOO_LARGE = "HTTP_RESPONSE_TOO_LARGE"
    HTTP_OUTPUT_ERROR = "HTTP_OUTPUT_ERROR"
    PYTHON_WORKER_ERROR = "PYTHON_WORKER_ERROR"
    PYTHON_TIMEOUT = "PYTHON_TIMEOUT"
    PYTHON_PROTOCOL_ERROR = "PYTHON_PROTOCOL_ERROR"
    PYTHON_EXCEPTION = "PYTHON_EXCEPTION"
    PYTHON_SERIALIZATION_ERROR = "PYTHON_SERIALIZATION_ERROR"
    LANGGRAPH_IMPORT_ERROR = "LANGGRAPH_IMPORT_ERROR"
    LANGGRAPH_RUNTIME_ERROR = "LANGGRAPH_RUNTIME_ERROR"
    LANGGRAPH_OUTPUT_ERROR = "LANGGRAPH_OUTPUT_ERROR"


class TargetConfigurationError(AgentQAError):
    """Raised when no safe adapter can be built for a target spec."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=TargetErrorCode.CONFIGURATION_ERROR.value)


class TargetSafetyError(AgentQAError):
    """Raised when an adapter request violates a security boundary."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=TargetErrorCode.HTTP_SAFETY_ERROR.value)
