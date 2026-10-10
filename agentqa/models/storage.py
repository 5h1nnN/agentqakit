"""Persistent result-store models kept separate from runtime evidence."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from agentqa.models.base import (
    NonEmptyString,
    NonNegativeInt,
    ResourceName,
    StrictModel,
    UtcDateTime,
)
from agentqa.models.results import RunResult, RunSummary


class StoredFinding(StrictModel):
    """One indexed violation suitable for review and filtering."""

    finding_id: NonEmptyString
    run_id: NonEmptyString
    test_case_id: NonEmptyString
    trial_id: NonEmptyString
    type: NonEmptyString
    message: NonEmptyString
    span_id: NonEmptyString | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class FindingReview(StrictModel):
    """Human review state stored independently from immutable evidence."""

    finding_id: NonEmptyString
    status: Literal["unreviewed", "confirmed", "false_positive", "resolved"]
    note: str = ""
    updated_at: UtcDateTime


class StoredRunSummary(StrictModel):
    """List-view metadata unavailable in the aggregate RunSummary alone."""

    project_id: ResourceName
    run_id: NonEmptyString
    status: Literal["COMPLETED", "FAILED", "CANCELLED"]
    exit_code: Literal[0, 1, 2]
    started_at: UtcDateTime
    ended_at: UtcDateTime
    summary: RunSummary
    mutation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    artifact_path: NonEmptyString | None = None
    finding_count: NonNegativeInt = 0


class StoredRun(StrictModel):
    """A persisted RunResult plus project and artifact location metadata."""

    project_id: ResourceName
    run: RunResult
    artifact_path: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_artifact_path(self) -> "StoredRun":
        if self.artifact_path is not None and not self.artifact_path.strip():
            msg = "artifact_path must be non-empty when present"
            raise ValueError(msg)
        return self

    def to_summary(self, *, finding_count: int) -> StoredRunSummary:
        score = None if self.run.mutation_score is None else self.run.mutation_score.overall.score
        return StoredRunSummary(
            project_id=self.project_id,
            run_id=self.run.run_id,
            status=self.run.status,
            exit_code=self.run.exit_code,
            started_at=self.run.started_at,
            ended_at=self.run.ended_at,
            summary=self.run.summary,
            mutation_score=score,
            artifact_path=self.artifact_path,
            finding_count=finding_count,
        )
