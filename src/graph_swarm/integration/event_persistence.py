"""Persist canonical agent events through the operational memory boundary."""

from graph_swarm.detection.failure_detector import detect_failure
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent
from graph_swarm.domain.failures import FailureEpisode
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph.repository import OperationalMemoryRepository


def persist_agent_event(
    repository: OperationalMemoryRepository,
    event: AgentEvent,
    task: Task,
    run: Run,
    environment: EnvironmentContext,
) -> FailureEpisode | None:
    """Persist one event's execution lineage and any detected failure."""
    _require_matching_identity(event, task, run)
    result = event.result
    repository.save_task(task)
    repository.save_run(run)
    repository.save_environment(environment)

    action = _planned_action(event, result)
    repository.save_action(action, result)
    repository.save_tool(Tool(name=result.tool_name))
    repository.link_task_action(task.id, action.id)
    repository.link_action_tool(action.id, result.tool_name)
    repository.link_action_run(action.id, run.id)

    failure = detect_failure(event)
    if failure is None:
        return None

    repository.save_failure(failure)
    repository.link_action_failure(action.id, failure.id)
    repository.link_failure_environment(failure.id, environment.id)
    return failure


def _planned_action(event: AgentEvent, result: ActionResult) -> PlannedAction:
    return PlannedAction(
        id=event.action_id,
        run_id=event.run_id,
        task_id=event.task_id,
        tool=result.tool_name,
        operation=result.tool_name,
        arguments={},
        planned_at=result.started_at,
    )


def _require_matching_identity(event: AgentEvent, task: Task, run: Run) -> None:
    if task.id != event.task_id:
        raise ValueError("Task.id must match AgentEvent.task_id")
    if run.id != event.run_id:
        raise ValueError("Run.id must match AgentEvent.run_id")
    if run.task_id != task.id:
        raise ValueError("Run.task_id must match Task.id")
