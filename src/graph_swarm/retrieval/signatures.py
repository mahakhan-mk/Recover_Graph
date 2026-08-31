"""Small helpers for deterministic structural recurrence keys."""

from graph_swarm.domain.actions import PlannedAction


def action_structure(action: PlannedAction) -> tuple[str, str]:
    """Return the exact tool/operation key used by graph retrieval."""
    return action.tool, action.operation


def normalize_signature(signature: str) -> str:
    """Normalize only whitespace/case; no semantic or embedding comparison."""
    return " ".join(signature.split()).casefold()
