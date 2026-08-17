"""Pre-execution advisory contracts."""

from pydantic import BaseModel, Field

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext


class EvaluateActionRequest(BaseModel):
    action: PlannedAction
    environment: EnvironmentContext


class AdviceResult(BaseModel):
    risk_level: str = "none"
    matched_incident_id: str | None = None
    applicability_score: float | None = None
    recovery_summary: str | None = None
    confidence: float | None = None
    provenance: list[str] = Field(default_factory=list)
