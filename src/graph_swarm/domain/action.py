"""Canonical action execution-result contract."""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, field_validator, model_validator


def _require_timezone_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class ActionResult(BaseModel):
    """Normalized result of one executed agent or tool action."""

    action_id: str
    tool_name: str
    success: bool
    exit_code: int | None = None
    output: str | None = None
    error: str | None = None
    started_at: datetime
    completed_at: datetime

    @field_validator("action_id", "tool_name")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be a non-empty string")
        return value

    @field_validator("started_at", "completed_at")
    @classmethod
    def require_aware_timestamps(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)

    @model_validator(mode="after")
    def require_chronological_timestamps(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not be before started_at")
        return self
