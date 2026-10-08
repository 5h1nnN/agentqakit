"""Shared Pydantic model foundations."""

from datetime import UTC, datetime
from typing import Annotated, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
)

from agentqa.conventions import to_utc_iso8601

API_VERSION = "agentqa.dev/v1alpha1"

NonEmptyString: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ResourceName: TypeAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
PositiveInt: TypeAlias = Annotated[int, Field(gt=0)]
NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]
UnitInterval: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]


def to_camel(value: str) -> str:
    """Convert an internal snake_case field name to lower camelCase."""
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        msg = "datetime must be timezone-aware"
        raise ValueError(msg)
    return value.astimezone(UTC)


UtcDateTime: TypeAlias = Annotated[
    datetime,
    AfterValidator(_normalize_utc),
    PlainSerializer(to_utc_iso8601, return_type=str, when_used="json"),
]


class StrictModel(BaseModel):
    """Base for internal models serialized with snake_case JSON fields."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class SpecModel(BaseModel):
    """Base for user YAML models with camelCase aliases."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        populate_by_name=True,
        strict=True,
        validate_assignment=True,
    )
