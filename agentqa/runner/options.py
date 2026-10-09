"""Sequential runner configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunnerOptions:
    """Timeouts, reproducibility seed, and artifact destination for a run."""

    artifact_directory: Path = Path("artifacts")
    random_seed: int = 0
    prepare_timeout_seconds: float = 30.0
    item_timeout_seconds: float = 120.0
    evaluator_timeout_seconds: float = 30.0
    cleanup_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        timeouts = {
            "prepare_timeout_seconds": self.prepare_timeout_seconds,
            "item_timeout_seconds": self.item_timeout_seconds,
            "evaluator_timeout_seconds": self.evaluator_timeout_seconds,
            "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
        }
        for name, value in timeouts.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
