"""Strict cassette and replay data contracts."""

from typing import Literal

from pydantic import JsonValue, model_validator

from agentqa.models.base import NonEmptyString, PositiveInt, StrictModel, UtcDateTime
from agentqa.specs.hashing import canonical_json_bytes, sha256_digest

CASSETTE_SCHEMA_VERSION: Literal["agentqa.cassette/v1alpha1"] = "agentqa.cassette/v1alpha1"


class CassetteError(StrictModel):
    """Safe exception evidence that can be replayed without importing its class."""

    code: NonEmptyString
    exception_type: NonEmptyString
    message: NonEmptyString


class CassetteEntry(StrictModel):
    """One canonical dependency interaction stored as a JSONL record."""

    schema_version: Literal["agentqa.cassette/v1alpha1"] = CASSETTE_SCHEMA_VERSION
    kind: Literal["tool", "retriever", "http"]
    name: NonEmptyString
    arguments: JsonValue
    call_index: PositiveInt
    status: Literal["success", "error"]
    result: JsonValue = None
    result_ref: NonEmptyString | None = None
    error: CassetteError | None = None
    recorded_at: UtcDateTime
    trace_id: NonEmptyString
    span_id: NonEmptyString
    checksum: NonEmptyString

    @model_validator(mode="after")
    def validate_outcome(self) -> "CassetteEntry":
        if self.status == "error" and self.error is None:
            msg = "error cassette entry requires error details"
            raise ValueError(msg)
        if self.status == "success" and self.error is not None:
            msg = "successful cassette entry cannot contain error details"
            raise ValueError(msg)
        if self.result_ref is not None and self.result is not None:
            msg = "cassette entry cannot contain both result and result_ref"
            raise ValueError(msg)
        return self

    def checksum_payload(self) -> bytes:
        """Return the canonical bytes covered by the entry checksum."""
        document = self.model_dump(mode="json")
        document.pop("checksum")
        return canonical_json_bytes(document)

    def verify_checksum(self) -> bool:
        """Verify this entry's embedded canonical checksum."""
        return self.checksum == sha256_digest(self.checksum_payload())
