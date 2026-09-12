"""Canonical contracts shared by the research experiment tracks."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from graph_swarm.domain._validation import require_non_empty
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import AdviceResult
from graph_swarm.domain.failures import FailureType


class ExperimentCondition(StrEnum):
    """Stable condition identifiers used across experiment configurations."""

    B0 = "B0"
    O1 = "O1"
    T = "T"


B0 = ExperimentCondition.B0
O1 = ExperimentCondition.O1
T = ExperimentCondition.T


EXPERIMENT_CONDITIONS: tuple[ExperimentCondition, ...] = (
    ExperimentCondition.B0,
    ExperimentCondition.O1,
    ExperimentCondition.T,
)


class ExperimentRunArtifact(BaseModel):
    """Typed, serializable record for one experiment run.

    The action, execution, and advice fields deliberately reference existing
    domain contracts so experiment recording cannot silently drift from the
    operational event semantics.
    """

    experiment_id: str
    condition: ExperimentCondition
    run_id: str
    task_id: str
    family_id: str
    chronological_index: int
    model: str
    model_settings: dict[str, object] = Field(default_factory=dict)
    prompt_version: str
    planned_action: PlannedAction
    executed_action: ActionResult
    advice_received: AdviceResult | None = None
    advice_accepted: bool = False
    failure_type: FailureType | None = None
    task_success: bool
    known_failure_repeated: bool
    tool_calls: int
    retries: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    retrieved_incident_id: str | None = None
    retrieval_score: float | None = None

    @field_validator(
        "experiment_id",
        "run_id",
        "task_id",
        "family_id",
        "model",
        "prompt_version",
    )
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        return require_non_empty(value)


# Short aliases keep the contract convenient for callers without introducing
# alternate schemas or duplicate models.
RunArtifact = ExperimentRunArtifact
PerRunArtifact = ExperimentRunArtifact


__all__ = [
    "EXPERIMENT_CONDITIONS",
    "B0",
    "ExperimentCondition",
    "ExperimentRunArtifact",
    "O1",
    "PerRunArtifact",
    "RunArtifact",
    "T",
]
