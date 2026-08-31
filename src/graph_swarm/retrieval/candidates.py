"""Focused read models returned by historical recovery retrieval."""

from datetime import datetime

from pydantic import BaseModel, field_validator

from graph_swarm.domain._validation import require_non_empty, require_timezone_aware
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution


class HistoricalActionContext(BaseModel):
    """Only the failed action fields needed for structural applicability."""

    id: str
    tool: str
    operation: str
    planned_at: datetime

    @field_validator("id", "tool", "operation")
    @classmethod
    def require_non_empty_fields(cls, value: str) -> str:
        return require_non_empty(value)

    @field_validator("planned_at")
    @classmethod
    def require_aware_planned_at(cls, value: datetime) -> datetime:
        return require_timezone_aware(value)


class HistoricalRecoveryCandidate(BaseModel):
    """Graph-layer read model for one eligible historical recovery.

    Neo4j nodes and records are deliberately converted to these domain models
    at the repository boundary.
    """

    failure: FailureEpisode
    failed_action: HistoricalActionContext
    environment: EnvironmentContext
    resolution: Resolution
    outcomes: tuple[Outcome, ...] = ()

    @property
    def successful_outcome_count(self) -> int:
        return sum(outcome.success for outcome in self.outcomes)
