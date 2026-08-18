"""Objective task/action outcome contracts."""

from datetime import datetime

from pydantic import BaseModel, field_validator

from graph_swarm.domain._validation import require_non_empty, require_timezone_aware


class Outcome(BaseModel):
    id: str
    action_id: str
    success: bool
    tests_passed: int | None = None
    tests_failed: int | None = None
    exit_code: int | None = None
    observed_at: datetime

    @field_validator("id", "action_id")
    @classmethod
    def require_non_empty_references(cls, value: str) -> str:
        return require_non_empty(value)

    @field_validator("observed_at")
    @classmethod
    def require_aware_observed_at(cls, value: datetime) -> datetime:
        return require_timezone_aware(value)
