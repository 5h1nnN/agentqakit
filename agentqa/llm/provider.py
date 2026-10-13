"""Shared contracts for bounded structured LLM operations."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import Field, JsonValue

from agentqa.conventions import new_id
from agentqa.models.base import NonEmptyString, NonNegativeInt, PositiveInt, StrictModel

LLMUseCase = Literal["test_generation", "semantic_judge", "failure_analysis"]
LLMCallStatus = Literal["SUCCEEDED", "INVALID_OUTPUT", "PROVIDER_ERROR", "SKIPPED"]


@dataclass(frozen=True, slots=True)
class StructuredRequest:
    """One bounded prompt with a strict machine-readable response contract."""

    use_case: LLMUseCase
    system_prompt: str
    prompt: str
    response_schema: dict[str, JsonValue]
    attempt: int = 1
    evidence_hash: str | None = None
    max_output_tokens: int = 1_024
    timeout_seconds: float = 60.0
    max_retries: int = 1
    redaction_policy: Literal["secrets"] = "secrets"
    request_id: str = field(default_factory=new_id)

    def __post_init__(self) -> None:
        if not 1 <= len(self.system_prompt) <= 32_000:
            raise ValueError("structured LLM system_prompt must contain 1 to 32000 characters")
        if not 1 <= len(self.prompt) <= 256_000:
            raise ValueError("structured LLM prompt must contain 1 to 256000 characters")
        schema_bytes = json.dumps(
            self.response_schema,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(schema_bytes) > 64 * 1024:
            raise ValueError("structured LLM response schema exceeds 65536 bytes")
        try:
            Draft202012Validator.check_schema(self.response_schema)
        except SchemaError:
            raise ValueError(
                "structured LLM response_schema must be valid Draft 2020-12 JSON Schema"
            ) from None
        if self.attempt not in {1, 2}:
            raise ValueError("structured LLM attempt must be 1 or 2")
        if self.evidence_hash is not None and not self.evidence_hash.strip():
            raise ValueError("structured LLM evidence_hash must be non-empty")
        if isinstance(self.max_output_tokens, bool) or not 64 <= self.max_output_tokens <= 16_384:
            raise ValueError("structured LLM max_output_tokens must be between 64 and 16384")
        if not 1.0 <= self.timeout_seconds <= 600.0:
            raise ValueError("structured LLM timeout_seconds must be between 1 and 600")
        if isinstance(self.max_retries, bool) or self.max_retries not in {0, 1}:
            raise ValueError("structured LLM max_retries must be 0 or 1")
        if not self.request_id.strip() or len(self.request_id) > 256:
            raise ValueError("structured LLM request_id must contain 1 to 256 characters")


@dataclass(frozen=True, slots=True)
class StructuredResponse:
    """Provider output plus safe request accounting."""

    content: str | Mapping[str, JsonValue]
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


class StructuredLLMProvider(Protocol):
    """Provider-neutral transport used by generators, judges, and analysts."""

    @property
    def name(self) -> NonEmptyString: ...

    @property
    def model(self) -> NonEmptyString: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    async def complete(self, request: StructuredRequest) -> StructuredResponse: ...


class LLMCallEvidence(StrictModel):
    """Non-secret, reproducible metadata for one logical structured LLM call."""

    use_case: LLMUseCase
    provider: NonEmptyString
    model: NonEmptyString
    parameters: dict[str, JsonValue]
    system_prompt_hash: NonEmptyString
    prompt_hash: NonEmptyString
    evidence_hash: NonEmptyString
    response_schema_hash: NonEmptyString
    request_id: NonEmptyString
    provider_request_ids: list[NonEmptyString] = Field(default_factory=list, max_length=10)
    token_usage: dict[str, NonNegativeInt] = Field(default_factory=dict, max_length=20)
    latency_ms: Annotated[float, Field(ge=0)] | None = None
    attempts: PositiveInt
    transport_attempts: PositiveInt | None = None
    status: LLMCallStatus
