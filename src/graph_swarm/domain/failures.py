"""Failure episode domain model."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, field_validator

from graph_swarm.domain._validation import require_non_empty, require_timezone_aware


class FailureType(StrEnum):
    TEST_FAILURE = "test_failure"
    COMMAND_FAILURE = "command_failure"
    DEPENDENCY_ERROR = "dependency_error"
    CONFIGURATION_ERROR = "configuration_error"
    TOOL_PARAMETER_ERROR = "tool_parameter_error"


class FailureEpisode(BaseModel):
    id: str
    action_id: str
    failure_type: FailureType
    signature: str
    symptom: str
    observed_at: datetime

    @field_validator("id", "action_id")
    @classmethod
    def require_non_empty_references(cls, value: str) -> str:
        return require_non_empty(value)

    @field_validator("observed_at")
    @classmethod
    def require_aware_observed_at(cls, value: datetime) -> datetime:
        return require_timezone_aware(value)
