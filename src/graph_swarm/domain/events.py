"""Typed event contracts emitted by agent and tool hooks."""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, field_validator, model_validator

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import HistoricalRecoveryAdvice


class AgentEventType(StrEnum):
    """Minimal Rollout 1 event taxonomy."""

    ACTION_COMPLETED = "action_completed"
    ADVICE_ISSUED = "advice_issued"


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


class AdviceEvent(BaseModel):
    """Canonical record that advisory context was issued before execution."""

    event_id: str
    run_id: str
    task_id: str
    event_type: AgentEventType = AgentEventType.ADVICE_ISSUED
    planned_action: PlannedAction
    advice: HistoricalRecoveryAdvice
    rendered_advice: str
    issued_at: datetime

    @field_validator("event_id", "run_id", "task_id", "rendered_advice")
    @classmethod
    def require_non_empty_advice_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("advice event fields must be non-empty strings")
        return value

    @field_validator("issued_at")
    @classmethod
    def require_aware_issued_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("issued_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def require_matching_action_context(self) -> Self:
        if self.planned_action.run_id != self.run_id:
            raise ValueError("advice planned action run_id must match event run_id")
        if self.planned_action.task_id != self.task_id:
            raise ValueError("advice planned action task_id must match event task_id")
        return self
