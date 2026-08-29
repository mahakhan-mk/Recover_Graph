"""Persist canonical agent events through the operational memory boundary."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from graph_swarm.detection.failure_detector import detect_failure
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent
from graph_swarm.domain.failures import FailureEpisode
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
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


def persist_agent_event_stream(
    repository: OperationalMemoryRepository,
    events: Sequence[AgentEvent],
    task: Task,
    run: Run,
    environment: EnvironmentContext,
) -> tuple[FailureEpisode, Resolution, Outcome] | None:
    """Persist GS-E001 recovery evidence from one ordered event stream.

    The resulting Resolution records observed sequence evidence only: a later
    successful write followed by a successful test run. It does not establish
    that the write caused the test result.
    """
    event_list = list(events)
    detected_failure: FailureEpisode | None = None
    failure_index: int | None = None
    for index, event in enumerate(event_list):
        failure = persist_agent_event(repository, event, task, run, environment)
        if failure is not None and detected_failure is None:
            detected_failure = failure
            failure_index = index

    if detected_failure is None or failure_index is None:
        return None

    write_index: int | None = None
    write_event: AgentEvent | None = None
    for index in range(failure_index + 1, len(event_list)):
        event = event_list[index]
        if event.result.tool_name == "write_file" and event.result.success:
            write_index = index
            write_event = event
            break

    if write_event is None or write_index is None:
        return None

    successful_test_event: AgentEvent | None = None
    for index in range(write_index + 1, len(event_list)):
        event = event_list[index]
        if (
            event.result.tool_name == "run_tests"
            and event.result.success
            and event.result.exit_code == 0
        ):
            successful_test_event = event
            break

    if successful_test_event is None:
        return None

    write_output = write_event.result.output
    description = (
        f"Observed recovery sequence: write_action_id={write_event.action_id}; "
    )
    if write_output is not None:
        description += f"write_output={write_output!r}; "
    description += (
        "successful_test_action_id="
        f"{successful_test_event.action_id}."
    )
    resolution = Resolution(
        id=str(uuid5(NAMESPACE_URL, f"graph-swarm/resolution/{detected_failure.id}")),
        failure_id=detected_failure.id,
        description=description,
        status=ResolutionStatus.OBSERVED_SUCCESSFUL,
        successful_observations=1,
        failed_observations=0,
    )
    successful_result = successful_test_event.result
    outcome = Outcome(
        id=str(uuid5(NAMESPACE_URL, f"graph-swarm/outcome/{successful_test_event.event_id}")),
        action_id=successful_test_event.action_id,
        success=True,
        exit_code=successful_result.exit_code,
        observed_at=successful_test_event.occurred_at,
    )
    repository.save_resolution(resolution)
    repository.link_failure_resolution(detected_failure.id, resolution.id)
    repository.save_outcome(outcome)
    repository.link_resolution_outcome(resolution.id, outcome.id)
    return detected_failure, resolution, outcome


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
