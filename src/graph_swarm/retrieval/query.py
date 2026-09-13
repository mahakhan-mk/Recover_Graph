"""Frozen semantic query construction for RecoveryPattern retrieval."""

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.tasks import Task

RECOVERY_RETRIEVAL_QUERY_VERSION = "v1"


def recovery_retrieval_query_text(
    task: Task,
    planned_action: PlannedAction,
    environment: EnvironmentContext,
) -> str:
    """Build the deterministic, repository-independent v1 retrieval text."""
    facts = [
        *(
            f"marker:{key}={_normalize(value)}"
            for key, value in sorted(environment.markers.items())
        ),
        *(
            f"version:{key}={_normalize(value)}"
            for key, value in sorted(environment.versions.items())
        ),
    ]
    environment_facts = "\n".join(facts) if facts else "(none)"
    return "\n".join(
        (
            "Task:",
            _normalize(task.problem_statement),
            "Planned tool: " + _normalize(planned_action.tool),
            "Planned operation: " + _normalize(planned_action.operation),
            "Runtime: " + _normalize(environment.runtime),
            "Environment:",
            environment_facts,
        )
    )


def _normalize(value: str) -> str:
    return " ".join(value.split())


__all__ = ["RECOVERY_RETRIEVAL_QUERY_VERSION", "recovery_retrieval_query_text"]
