"""Action-domain contracts."""

from datetime import datetime

from pydantic import BaseModel, Field

from graph_swarm.domain.action import ActionResult


class PlannedAction(BaseModel):
    """A tool action proposed by the experimental agent."""

    id: str
    run_id: str
    task_id: str
    tool: str
    operation: str
    arguments: dict[str, object] = Field(default_factory=dict)
    planned_at: datetime


__all__ = ["ActionResult", "PlannedAction"]
