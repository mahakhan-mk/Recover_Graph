"""Operational environment contracts used for recovery applicability."""

from pydantic import BaseModel, Field, field_validator

from graph_swarm.domain._validation import require_non_empty


class EnvironmentContext(BaseModel):
    id: str
    repository: str
    runtime: str
    versions: dict[str, str] = Field(default_factory=dict)
    markers: dict[str, str] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def require_non_empty_id(cls, value: str) -> str:
        return require_non_empty(value)
