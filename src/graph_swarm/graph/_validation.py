"""Validation helpers for the repository boundary."""

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction


def require_non_empty_id(value: str, field_name: str = "id") -> str:
    """Reject empty identifiers before a repository relationship write."""
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def validate_relationship_ids(**identifiers: str) -> None:
    """Validate all stable identities supplied to one relationship write."""
    for field_name, value in identifiers.items():
        require_non_empty_id(value, field_name)


def validate_action_persistence(action: PlannedAction, result: ActionResult) -> None:
    """Require a planned action and result to refer to the same action."""
    if action.id != result.action_id:
        raise ValueError("PlannedAction.id must match ActionResult.action_id")
