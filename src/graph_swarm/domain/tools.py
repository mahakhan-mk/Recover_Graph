"""Tool identity contract for action lineage."""

from pydantic import BaseModel, field_validator

from graph_swarm.domain._validation import require_non_empty


class Tool(BaseModel):
    """A tool identified by its stable Rollout 1 name."""

    name: str

    @field_validator("name")
    @classmethod
    def require_non_empty_name(cls, value: str) -> str:
        return require_non_empty(value)
