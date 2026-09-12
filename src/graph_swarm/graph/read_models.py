"""Typed, read-only-shaped models for reconstructed graph lineage."""

from pydantic import BaseModel, model_validator

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.recovery_patterns import RecoveryPattern
from graph_swarm.domain.resolutions import Resolution
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph._validation import validate_action_persistence


class ActionLineageRecord(BaseModel):
    """Read representation pairing one planned action with its result."""

    planned_action: PlannedAction
    result: ActionResult

    @model_validator(mode="after")
    def require_matching_action_id(self) -> "ActionLineageRecord":
        validate_action_persistence(self.planned_action, self.result)
        return self


class IncidentLineage(BaseModel):
    """Typed reconstruction of one failure and its Rollout 1 graph context."""

    run: Run
    task: Task
    actions: tuple[ActionLineageRecord, ...]
    tools: tuple[Tool, ...]
    failure: FailureEpisode
    environment: EnvironmentContext
    resolutions: tuple[Resolution, ...]
    outcomes: tuple[Outcome, ...]
    recovery_actions: tuple[ActionLineageRecord, ...] = ()

    @property
    def observed_changes(self) -> tuple[ActionLineageRecord, ...]:
        """Compatibility-friendly name for actions observed from resolutions."""
        return self.recovery_actions


class RecoveryPatternTask(BaseModel):
    """Pattern-audit task projection without benchmark family metadata."""

    id: str
    problem_statement: str
    repository: str
    chronological_index: int


class RecoveryPatternLineage(BaseModel):
    """Typed RecoveryPattern reconstruction with complete source evidence."""

    pattern: RecoveryPattern
    failure: FailureEpisode
    resolution: Resolution
    outcome: Outcome
    task: RecoveryPatternTask
    environment: EnvironmentContext
    recovery_action: ActionLineageRecord
