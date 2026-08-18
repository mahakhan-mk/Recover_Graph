"""Benchmark task contract used by Rollout 1 lineage."""

from pydantic import BaseModel, field_validator

from graph_swarm.domain._validation import require_non_empty


class Task(BaseModel):
    """Stable task identity and frozen pilot ordering metadata."""

    id: str
    family_id: str
    repository: str
    chronological_index: int

    @field_validator("id", "family_id", "repository")
    @classmethod
    def require_non_empty_identity_fields(cls, value: str) -> str:
        return require_non_empty(value)
