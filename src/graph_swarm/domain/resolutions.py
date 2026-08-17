"""Recovery evidence and verification state."""

from enum import StrEnum

from pydantic import BaseModel


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
