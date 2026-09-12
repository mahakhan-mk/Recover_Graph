"""Typed Recovery Memory V2 pattern contracts."""

from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from graph_swarm.domain._validation import require_non_empty, require_timezone_aware


class RecoveryPatternStatus(StrEnum):
    """The only verification states used by Recovery Memory V2."""

    CANDIDATE = "candidate"
    OBSERVED_SUCCESSFUL = "observed_successful"
    VERIFIED_FOR_EXPERIMENT = "verified_for_experiment"
    STALE = "stale"
    INVALIDATED = "invalidated"


class EnvironmentConstraints(BaseModel):
    """Applicability-relevant environment facts without repository matching."""

    model_config = ConfigDict(extra="forbid")

    runtime: str | None = None
    versions: dict[str, str] = Field(default_factory=dict)
    dependencies: dict[str, str] = Field(default_factory=dict)
    markers: dict[str, str] = Field(default_factory=dict)

    @field_validator("runtime")
    @classmethod
    def require_non_empty_runtime(cls, value: str | None) -> str | None:
        if value is not None:
            return require_non_empty(value)
        return value


class RecoveryPattern(BaseModel):
    """Historical operational evidence, not a causal explanation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    guidance: str
    source_failure_id: str
    source_resolution_id: str
    source_outcome_id: str
    source_task_id: str
    source_chronological_index: int
    source_tool: str
    source_operation: str
    source_failure_type: str
    environment_constraints: EnvironmentConstraints = Field(
        default_factory=EnvironmentConstraints
    )
    verification_status: RecoveryPatternStatus = RecoveryPatternStatus.CANDIDATE
    evidence_count: int
    evidence_summary: str
    embedding: list[float] | None = None
    created_at: datetime
    invalidated_at: datetime | None = None

    @field_validator(
        "id",
        "title",
        "guidance",
        "source_failure_id",
        "source_resolution_id",
        "source_outcome_id",
        "source_task_id",
        "source_tool",
        "source_operation",
        "source_failure_type",
        "evidence_summary",
    )
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        return require_non_empty(value)

    @field_validator("source_chronological_index", "evidence_count")
    @classmethod
    def require_non_negative_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("created_at", "invalidated_at")
    @classmethod
    def require_aware_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            return require_timezone_aware(value)
        return value

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("embedding must be a numeric list")
        values = cast(list[object], value)
        for item in values:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("embedding values must be numeric and not boolean")
            if not isfinite(float(item)):
                raise ValueError("embedding values must be finite")
        return values


__all__ = [
    "EnvironmentConstraints",
    "RecoveryPattern",
    "RecoveryPatternStatus",
]
