"""Persist canonical agent events through the operational memory boundary."""

from collections.abc import Mapping, Sequence
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
from graph_swarm.memory.recovery_evidence import (
    ConcreteRecoveryEvidence,
    arguments_json,
    has_concrete_change,
    normalize_planned_action,
)

PlannedActionContext = Mapping[str, PlannedAction] | Sequence[PlannedAction]


class MissingTrustedPlannedActionError(ValueError):
    """Raised when a T memory path has an event without runtime provenance."""

    code = "BLOCKED_MISSING_TRUSTED_PLANNED_ACTION_PROVENANCE"

    def __init__(self, action_ids: Sequence[str]) -> None:
        self.action_ids = tuple(action_ids)
        super().__init__(
            f"{self.code}: missing runtime PlannedAction for action IDs "
            f"{', '.join(self.action_ids)}"
        )


def persist_agent_event(
    repository: OperationalMemoryRepository,
    event: AgentEvent,
    task: Task,
    run: Run,
    environment: EnvironmentContext,
    *,
    planned_action: PlannedAction | None = None,
) -> FailureEpisode | None:
    """Persist one event's execution lineage and any detected failure."""
    _require_matching_identity(event, task, run)
    result = event.result
    repository.save_task(task)
    repository.save_run(run)
    repository.save_environment(environment)

    action = _planned_action(event, result, planned_action)
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
    *,
    planned_actions: PlannedActionContext | None = None,
    require_trusted_planned_actions: bool = False,
) -> tuple[FailureEpisode, Resolution, Outcome] | None:
    """Persist GS-E001 recovery evidence from one ordered event stream.

    The resulting Resolution records observed sequence evidence only: a later
    concrete change followed by a successful objective test run. It does not
    establish that the change caused the test result.
    """
    event_list = list(events)
    action_context = _index_planned_actions(planned_actions)
    if require_trusted_planned_actions:
        missing_action_ids = tuple(
            event.action_id for event in event_list if event.action_id not in action_context
        )
        if missing_action_ids:
            raise MissingTrustedPlannedActionError(missing_action_ids)
    detected_failure: FailureEpisode | None = None
    failure_index: int | None = None
    for index, event in enumerate(event_list):
        failure = persist_agent_event(
            repository,
            event,
            task,
            run,
            environment,
            planned_action=action_context.get(event.action_id),
        )
        if failure is not None and detected_failure is None:
            detected_failure = failure
            failure_index = index

    if detected_failure is None or failure_index is None:
        return None

    change_index: int | None = None
    change_event: AgentEvent | None = None
    change_action: PlannedAction | None = None
    for index in range(failure_index + 1, len(event_list)):
        event = event_list[index]
        action = action_context.get(event.action_id)
        if (
            event.result.success
            and action is not None
            and has_concrete_change(action)
        ):
            change_index = index
            change_event = event
            change_action = action
            break

    if change_event is None or change_index is None or change_action is None:
        return None

    successful_test_event: AgentEvent | None = None
    for index in range(change_index + 1, len(event_list)):
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

    normalized_change = normalize_planned_action(change_action)
    resolution_id = str(
        uuid5(NAMESPACE_URL, f"graph-swarm/resolution/{detected_failure.id}")
    )
    outcome_id = str(
        uuid5(NAMESPACE_URL, f"graph-swarm/outcome/{successful_test_event.event_id}")
    )
    evidence = ConcreteRecoveryEvidence(
        recovery_action=normalized_change,
        source_failure_id=detected_failure.id,
        resolution_id=resolution_id,
        objective_outcome_id=outcome_id,
        task_id=task.id,
        source_chronological_index=task.chronological_index,
        environment_id=environment.id,
    )
    description = (
        "Observed concrete recovery action: "
        f"action_id={evidence.recovery_action.id}; "
        f"run_id={evidence.recovery_action.run_id}; "
        f"task_id={evidence.task_id}; "
        f"tool={evidence.recovery_action.tool}; "
        f"operation={evidence.recovery_action.operation}; "
        f"arguments_json={arguments_json(evidence.recovery_action)}; "
        f"planned_at={evidence.recovery_action.planned_at.isoformat()}; "
        f"source_failure_id={evidence.source_failure_id}; "
        f"environment_id={evidence.environment_id}; "
        f"source_chronological_index={evidence.source_chronological_index}; "
        "objective_success_action_id="
        f"{successful_test_event.action_id}."
    )
    resolution = Resolution(
        id=resolution_id,
        failure_id=detected_failure.id,
        description=description,
        status=ResolutionStatus.OBSERVED_SUCCESSFUL,
        successful_observations=1,
        failed_observations=0,
        observed_at=successful_test_event.occurred_at,
    )
    successful_result = successful_test_event.result
    outcome = Outcome(
        id=outcome_id,
        action_id=successful_test_event.action_id,
        success=True,
        exit_code=successful_result.exit_code,
        observed_at=successful_test_event.occurred_at,
    )
    repository.save_resolution(resolution)
    repository.link_failure_resolution(detected_failure.id, resolution.id)
    repository.save_outcome(outcome)
    repository.link_resolution_outcome(resolution.id, outcome.id)
    repository.link_resolution_observed_change(
        resolution.id,
        evidence.recovery_action.id,
    )
    return detected_failure, resolution, outcome


def _planned_action(
    event: AgentEvent,
    result: ActionResult,
    planned_action: PlannedAction | None,
) -> PlannedAction:
    if planned_action is not None:
        if planned_action.id != event.action_id:
            raise ValueError("planned action id must match AgentEvent.action_id")
        if planned_action.run_id != event.run_id:
            raise ValueError("planned action run_id must match AgentEvent.run_id")
        if planned_action.task_id != event.task_id:
            raise ValueError("planned action task_id must match AgentEvent.task_id")
        if planned_action.tool != result.tool_name:
            raise ValueError("planned action tool must match ActionResult.tool_name")
        return normalize_planned_action(planned_action)

    # Legacy event callers can still persist execution lineage. They cannot
    # produce a recovery Resolution without a concrete action context.
    return PlannedAction(
        id=event.action_id,
        run_id=event.run_id,
        task_id=event.task_id,
        tool=result.tool_name,
        operation=result.tool_name,
        arguments={},
        planned_at=result.started_at,
    )


def _index_planned_actions(
    planned_actions: PlannedActionContext | None,
) -> dict[str, PlannedAction]:
    if planned_actions is None:
        return {}
    indexed: dict[str, PlannedAction] = {}
    if isinstance(planned_actions, Mapping):
        values = planned_actions.items()
    else:
        values = ((None, action) for action in planned_actions)
    for supplied_id, action in values:
        normalized_action = normalize_planned_action(action)
        if supplied_id is not None and supplied_id != normalized_action.id:
            raise ValueError(
                "planned action mapping key must match PlannedAction.id: "
                f"{supplied_id!r} != {normalized_action.id!r}"
            )
        existing = indexed.get(normalized_action.id)
        if existing is not None:
            if existing != normalized_action:
                raise ValueError(
                    "duplicate planned action id has different content: "
                    f"{normalized_action.id}"
                )
            continue
        indexed[normalized_action.id] = normalized_action
    return indexed


def _require_matching_identity(event: AgentEvent, task: Task, run: Run) -> None:
    if task.id != event.task_id:
        raise ValueError("Task.id must match AgentEvent.task_id")
    if run.id != event.run_id:
        raise ValueError("Run.id must match AgentEvent.run_id")
    if run.task_id != task.id:
        raise ValueError("Run.task_id must match Task.id")
