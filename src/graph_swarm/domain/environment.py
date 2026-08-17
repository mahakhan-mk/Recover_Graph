"""Operational environment contracts used for recovery applicability."""

from pydantic import BaseModel, Field


class EnvironmentContext(BaseModel):
    repository: str
    runtime: str
    versions: dict[str, str] = Field(default_factory=dict)
    markers: dict[str, str] = Field(default_factory=dict)
