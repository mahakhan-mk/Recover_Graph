"""Evidence for comparing planned behavior before advice with later action."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, field_validator, model_validator

from graph_swarm.domain._validation import require_non_empty, require_timezone_aware
from graph_swarm.domain.actions import PlannedAction


class BehaviorChangeEvidence(BaseModel):
    """Direct structural evidence, without an acceptance or success inference."""

    advice_event_id: str
    planned_action_before_advice: PlannedAction
    actual_action_after_advice: PlannedAction | None = None
    behavior_changed: bool | None = None
    observation: Literal["changed", "unchanged", "no_subsequent_action"]
    observed_at: datetime

    @field_validator("advice_event_id")
    @classmethod
    def require_non_empty_event_id(cls, value: str) -> str:
        return require_non_empty(value)

    @field_validator("observed_at")
    @classmethod
    def require_aware_observed_at(cls, value: datetime) -> datetime:
        return require_timezone_aware(value)

    @model_validator(mode="after")
    def require_consistent_observation(self) -> Self:
        if self.observation == "no_subsequent_action":
            if self.actual_action_after_advice is not None or self.behavior_changed is not None:
                raise ValueError("no_subsequent_action cannot contain a later action")
        elif self.actual_action_after_advice is None or self.behavior_changed is None:
            raise ValueError("observed behavior must contain a later action and comparison")
        return self
