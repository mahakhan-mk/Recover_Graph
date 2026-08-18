"""Run contract for one chronological task execution."""

from datetime import datetime

from pydantic import BaseModel, field_validator

from graph_swarm.domain._validation import require_non_empty, require_timezone_aware


class Run(BaseModel):
    """Identity and start time for one task execution run."""

    id: str
    task_id: str
    started_at: datetime

    @field_validator("id", "task_id")
    @classmethod
    def require_non_empty_ids(cls, value: str) -> str:
        return require_non_empty(value)

    @field_validator("started_at")
    @classmethod
    def require_aware_started_at(cls, value: datetime) -> datetime:
        return require_timezone_aware(value)
