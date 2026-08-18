"""Synchronous persistence boundary for operational memory."""

from typing import Protocol

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph.read_models import IncidentLineage


class OperationalMemoryRepository(Protocol):
    """Rollout 1 storage contract implemented by a synchronous repository.

    Entity saves are logically idempotent by stable identity. Relationship
    writes are logically idempotent by their endpoint identities. Concrete
    implementations must use MERGE-equivalent behavior rather than blind
    duplicate creation. Successful writes return ``None``; persistence,
    connection, and consistency failures propagate as exceptions. Relationship
    implementations must reject empty endpoint identities, using the shared
    repository-boundary validator where appropriate.
    """

    def save_run(self, run: Run) -> None:
        """Persist one Run by ``run.id`` using idempotent semantics."""
        ...

    def save_task(self, task: Task) -> None:
        """Persist one Task by ``task.id`` using idempotent semantics."""
        ...

    def save_action(self, action: PlannedAction, result: ActionResult) -> None:
        """Persist matching planned and result contracts as one Action node.

        Implementations must reject mismatched ``action.id`` and
        ``result.action_id`` values before writing.
        """
        ...

    def save_tool(self, tool: Tool) -> None:
        """Persist one Tool by ``tool.name`` using idempotent semantics."""
        ...

    def save_environment(self, environment: EnvironmentContext) -> None:
        """Persist one Environment by ``environment.id`` idempotently."""
        ...

    def save_failure(self, failure: FailureEpisode) -> None:
        """Persist one FailureEpisode by ``failure.id`` idempotently."""
        ...

    def save_resolution(self, resolution: Resolution) -> None:
        """Persist one Resolution by ``resolution.id`` idempotently."""
        ...

    def save_outcome(self, outcome: Outcome) -> None:
        """Persist one Outcome by ``outcome.id`` idempotently."""
        ...

    def link_task_action(self, task_id: str, action_id: str) -> None:
        """Create ``Task-HAS_ACTION->Action`` idempotently."""
        ...

    def link_action_tool(self, action_id: str, tool_name: str) -> None:
        """Create ``Action-USED->Tool`` idempotently."""
        ...

    def link_action_run(self, action_id: str, run_id: str) -> None:
        """Create ``Action-PART_OF->Run`` idempotently."""
        ...

    def link_action_failure(self, action_id: str, failure_id: str) -> None:
        """Create ``Action-PART_OF_FAILURE->FailureEpisode`` idempotently."""
        ...

    def link_failure_environment(self, failure_id: str, environment_id: str) -> None:
        """Create ``FailureEpisode-OCCURRED_IN->Environment`` idempotently."""
        ...

    def link_failure_resolution(self, failure_id: str, resolution_id: str) -> None:
        """Create ``FailureEpisode-RESOLVED_BY->Resolution`` idempotently."""
        ...

    def link_resolution_outcome(self, resolution_id: str, outcome_id: str) -> None:
        """Create ``Resolution-VERIFIED_BY->Outcome`` idempotently."""
        ...

    def get_incident_lineage(self, failure_id: str) -> IncidentLineage:
        """Return the typed Rollout 1 lineage rooted at ``failure_id``."""
        ...
