"""Benchmark task contract used by Rollout 1 lineage."""

from pydantic import BaseModel, field_validator

from graph_swarm.domain._validation import require_non_empty


class Task(BaseModel):
    """Agent-visible task context plus frozen pilot ordering metadata.

    ``family_id`` and ``chronological_index`` are research metadata. They are
    carried for persistence and analysis but are not part of the task context
    presented to the agent.
    """

    id: str
    problem_statement: str
    family_id: str
    repository: str
    chronological_index: int

    @field_validator("id", "problem_statement", "family_id", "repository")
    @classmethod
    def require_non_empty_identity_fields(cls, value: str) -> str:
        return require_non_empty(value)
