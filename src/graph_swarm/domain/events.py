"""Typed event contracts emitted by agent and tool hooks."""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, field_validator, model_validator

from graph_swarm.domain.action import ActionResult


class AgentEventType(StrEnum):
    """Minimal Rollout 1 event taxonomy."""

    ACTION_COMPLETED = "action_completed"


class AgentEvent(BaseModel):
    """Normalized event emitted after an agent action produces a result."""

    event_id: str
    run_id: str
    task_id: str
    action_id: str
    event_type: AgentEventType
    result: ActionResult
    occurred_at: datetime

    @field_validator("event_id", "run_id", "task_id", "action_id")
    @classmethod
    def require_non_empty_ids(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("IDs must be non-empty strings")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def require_matching_action_id(self) -> Self:
        if self.action_id != self.result.action_id:
            raise ValueError("event action_id must match result action_id")
        return self
