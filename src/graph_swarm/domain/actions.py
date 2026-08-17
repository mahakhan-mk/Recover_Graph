"""Action-domain contracts."""

from datetime import datetime

from pydantic import BaseModel, Field


class PlannedAction(BaseModel):
    """A tool action proposed by the experimental agent."""

    id: str
    run_id: str
    task_id: str
    tool: str
    operation: str
    arguments: dict[str, object] = Field(default_factory=dict)
    planned_at: datetime


class ActionResult(BaseModel):
    """Normalized result of a tool action."""

    action_id: str
    status: str
    exit_code: int | None = None
    http_status: int | None = None
    exception_class: str | None = None
    error: str | None = None
