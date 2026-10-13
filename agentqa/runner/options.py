"""Test runner configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentqa.conventions import is_uuid


@dataclass(frozen=True, slots=True)
class RunnerOptions:
    """Timeouts, reproducibility seed, and artifact destination for a run."""

    artifact_directory: Path = Path("artifacts")
    random_seed: int = 0
    concurrency: int = 1
    max_total_tokens: int | None = None
    max_total_cost_usd: float | None = None
    prepare_timeout_seconds: float = 30.0
    item_timeout_seconds: float = 120.0
    evaluator_timeout_seconds: float = 30.0
    cleanup_timeout_seconds: float = 30.0
    trace_flush_timeout_seconds: float = 0.25
    trace_quiet_period_seconds: float = 0.05
    otlp_endpoint: str | None = None
    replay_mode: Literal["off", "record", "replay"] = "record"
    replay_cassette: Path | None = None
    allow_live_fallback: bool = False
    replay_volatile_fields: tuple[str, ...] = ()
    replay_max_inline_result_bytes: int = 65_536
    replay_trial_seed: int | None = None
    generate_failure_bundles: bool = True
    original_command: str | None = None
    run_id: str | None = None
    manifest_workspace: Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.concurrency, bool) or self.concurrency <= 0:
            raise ValueError("concurrency must be a positive integer")
        if self.max_total_tokens is not None and (
            isinstance(self.max_total_tokens, bool) or self.max_total_tokens <= 0
        ):
            raise ValueError("max_total_tokens must be a positive integer")
        if self.max_total_cost_usd is not None and (
            isinstance(self.max_total_cost_usd, bool) or self.max_total_cost_usd <= 0
        ):
            raise ValueError("max_total_cost_usd must be positive")
        timeouts = {
            "prepare_timeout_seconds": self.prepare_timeout_seconds,
            "item_timeout_seconds": self.item_timeout_seconds,
            "evaluator_timeout_seconds": self.evaluator_timeout_seconds,
            "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
            "trace_flush_timeout_seconds": self.trace_flush_timeout_seconds,
            "trace_quiet_period_seconds": self.trace_quiet_period_seconds,
        }
        for name, value in timeouts.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.replay_mode not in {"off", "record", "replay"}:
            raise ValueError("replay_mode must be off, record, or replay")
        if self.replay_mode == "replay" and self.replay_cassette is None:
            raise ValueError("replay mode requires replay_cassette")
        if self.replay_mode != "replay" and self.allow_live_fallback:
            raise ValueError("allow_live_fallback is only valid in replay mode")
        if self.replay_max_inline_result_bytes <= 0:
            raise ValueError("replay_max_inline_result_bytes must be positive")
        if any(not path.strip() for path in self.replay_volatile_fields):
            raise ValueError("replay volatile field paths must be non-empty")
        if self.run_id is not None and not is_uuid(self.run_id):
            raise ValueError("run_id must be a canonical UUID")
