"""Recovery evidence and verification state."""

from enum import StrEnum

from pydantic import BaseModel, field_validator

from graph_swarm.domain._validation import require_non_empty


class ResolutionStatus(StrEnum):
    CANDIDATE = "candidate"
    OBSERVED_SUCCESSFUL = "observed_successful"
    VERIFIED = "verified"
    STALE = "stale"


class Resolution(BaseModel):
    id: str
    failure_id: str
    description: str
    status: ResolutionStatus = ResolutionStatus.CANDIDATE
    successful_observations: int = 0
    failed_observations: int = 0

    @field_validator("id", "failure_id")
    @classmethod
    def require_non_empty_references(cls, value: str) -> str:
        return require_non_empty(value)
