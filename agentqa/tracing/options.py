"""Trace collection resource and privacy limits."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TraceOptions:
    """Bound trace capture before data reaches a store or artifact."""

    flush_timeout_seconds: float = 0.25
    quiet_period_seconds: float = 0.05
    max_spans_per_trace: int = 2_000
    max_attributes_per_span: int = 128
    max_attribute_chars: int = 4_096
    max_content_bytes: int = 16_384
    max_receiver_bytes: int = 4_194_304
    capture_content: bool = False

    def __post_init__(self) -> None:
        positive_values = {
            "flush_timeout_seconds": self.flush_timeout_seconds,
            "quiet_period_seconds": self.quiet_period_seconds,
            "max_spans_per_trace": self.max_spans_per_trace,
            "max_attributes_per_span": self.max_attributes_per_span,
            "max_attribute_chars": self.max_attribute_chars,
            "max_content_bytes": self.max_content_bytes,
            "max_receiver_bytes": self.max_receiver_bytes,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
