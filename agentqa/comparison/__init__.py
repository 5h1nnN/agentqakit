"""Run result comparison."""

from agentqa.comparison.artifacts import (
    ComparisonError,
    RunArtifact,
    load_run_artifact,
    metrics_for,
)
from agentqa.comparison.core import compare_runs
from agentqa.comparison.io import write_report

__all__ = [
    "ComparisonError",
    "RunArtifact",
    "compare_runs",
    "load_run_artifact",
    "metrics_for",
    "write_report",
]
