"""Failure episode domain model."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


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
