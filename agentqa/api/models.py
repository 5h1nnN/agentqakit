"""Strict HTTP request and asynchronous job models."""

from typing import Literal

from pydantic import ConfigDict, Field

from agentqa.models.base import (
    NonEmptyString,
    PositiveInt,
    ResourceName,
    StrictModel,
    UtcDateTime,
    to_camel,
)
from agentqa.models.results import ErrorInfo


class ApiModel(StrictModel):
    """HTTP model accepting and returning lower-camel aliases."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        populate_by_name=True,
        strict=True,
    )


class RunCreateRequest(ApiModel):
    """Workspace-relative file request for one local background run."""

    project_id: ResourceName = "default"
    tests_path: NonEmptyString
    target_path: NonEmptyString | None = None
    mutations_path: NonEmptyString | None = None
    manifest_path: NonEmptyString | None = None
    random_seed: int = 0
    concurrency: PositiveInt = 1
    max_total_tokens: PositiveInt | None = None
    max_total_cost_usd: float | None = Field(default=None, gt=0)
    capture_trace_content: bool = False
    generate_failure_bundles: bool = True
    judge_enabled: bool = False


class RunJob(ApiModel):
    """Observable lifecycle for a background API execution."""

    run_id: NonEmptyString
    project_id: ResourceName
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
    submitted_at: UtcDateTime
    finished_at: UtcDateTime | None = None
    error: ErrorInfo | None = None


class ComparisonRequest(ApiModel):
    """Stored run identifiers to compare."""

    baseline_run_id: NonEmptyString
    candidate_run_id: NonEmptyString


class GateEvaluateRequest(ApiModel):
    """Stored candidate, optional baseline, and workspace-relative gate spec."""

    gate_path: NonEmptyString
    candidate_run_id: NonEmptyString
    baseline_run_id: NonEmptyString | None = None


class ReviewUpdateRequest(ApiModel):
    """Mutable human judgement over one immutable finding."""

    status: Literal["unreviewed", "confirmed", "false_positive", "resolved"]
    note: str = Field(default="", max_length=10_000)
