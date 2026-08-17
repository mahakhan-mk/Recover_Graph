"""Typed event contracts emitted by agent and tool hooks."""

from datetime import datetime

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    """Framework-neutral event captured during an experimental run."""

    id: str
    run_id: str
    task_id: str
    action_id: str | None = None
    event_type: str
    timestamp: datetime
    payload: dict[str, object] = Field(default_factory=dict)
