"""Private protocol models for the Python SDK subprocess bridge."""

from typing import Literal

from pydantic import Field

from agentqa.models import (
    MutationApplication,
    MutationSpec,
    ReplayStats,
    RunContext,
    TraceEvent,
)
from agentqa.models.base import PositiveInt, StrictModel
from agentqa.replay import CassetteEntry


class PythonSDKReplayConfig(StrictModel):
    mode: Literal["off", "record", "replay"]
    source_path: str | None = None
    output_path: str | None = None
    max_inline_result_bytes: PositiveInt = 65_536
    allow_live_fallback: bool = False
    volatile_fields: list[str] = Field(default_factory=list)


class PythonSDKTraceConfig(StrictModel):
    max_spans_per_trace: PositiveInt = 2_000


class PythonSDKTrialConfig(StrictModel):
    context: RunContext
    mutations: list[MutationSpec] = Field(default_factory=list)
    replay: PythonSDKReplayConfig
    trace: PythonSDKTraceConfig = Field(default_factory=PythonSDKTraceConfig)


class PythonSDKTrialEvidence(StrictModel):
    mutation_applications: list[MutationApplication] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    replay: ReplayStats
    cassette_entries: list[CassetteEntry] = Field(default_factory=list)


__all__ = [
    "PythonSDKReplayConfig",
    "PythonSDKTraceConfig",
    "PythonSDKTrialConfig",
    "PythonSDKTrialEvidence",
]
