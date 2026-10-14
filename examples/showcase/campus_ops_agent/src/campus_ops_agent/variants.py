"""Intentional defect switches kept independent for controlled experiments."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

Variant: TypeAlias = Literal[
    "bug-retry",
    "bug-confirmation",
    "bug-authorization",
    "bug-injection",
    "bug-idempotency",
    "fixed",
]


@dataclass(frozen=True, slots=True)
class VariantConfig:
    name: Variant
    max_retries: int | None = 2
    require_confirmation: bool = True
    authorize_before_tool: bool = True
    trust_retrieved_instructions: bool = False
    enforce_idempotency: bool = True


VARIANTS: dict[Variant, VariantConfig] = {
    "bug-retry": VariantConfig(name="bug-retry", max_retries=None),
    "bug-confirmation": VariantConfig(name="bug-confirmation", require_confirmation=False),
    "bug-authorization": VariantConfig(name="bug-authorization", authorize_before_tool=False),
    "bug-injection": VariantConfig(name="bug-injection", trust_retrieved_instructions=True),
    "bug-idempotency": VariantConfig(name="bug-idempotency", enforce_idempotency=False),
    "fixed": VariantConfig(name="fixed"),
}


def get_variant(name: Variant | str) -> VariantConfig:
    try:
        return VARIANTS[name]  # type: ignore[index]
    except KeyError as exc:
        choices = ", ".join(VARIANTS)
        raise ValueError(f"unknown variant {name!r}; choose one of: {choices}") from exc
