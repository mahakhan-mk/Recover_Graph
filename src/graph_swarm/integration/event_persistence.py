"""Persist canonical agent events through the operational memory boundary."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from graph_swarm.detection.failure_detector import detect_failure
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.memory.recovery_evidence import (
    ConcreteRecoveryEvidence,
    RepositoryMutationEvidence,
    arguments_json,
    has_concrete_change,
    is_test_execution,
    normalize_planned_action,
)

PlannedActionContext = Mapping[str, PlannedAction] | Sequence[PlannedAction]


@dataclass(frozen=True)
class RecoveryEventChain:
    """Earliest complete observational recovery sequence."""

    failure_event: AgentEvent
    failure: FailureEpisode
    change_event: AgentEvent
    change_action: PlannedAction
    verification_event: AgentEvent


@dataclass(frozen=True)
class ObjectiveAnchor:
    """Typed, out-of-band observation of one frozen objective evaluation."""

    run_id: str
    task_id: str
    passed: bool
    return_code: int | None
    status: str
    started_at: datetime
    completed_at: datetime

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.task_id.strip() or not self.status.strip():
            raise ValueError("objective anchors require run and task IDs")
        for value in (self.started_at, self.completed_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("objective anchor timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("objective anchor completed_at must not precede started_at")


def select_recovery_event_chain(
    events: Sequence[AgentEvent],
    planned_actions: Mapping[str, PlannedAction],
    mutation_evidence: Mapping[str, RepositoryMutationEvidence] | None = None,
) -> RecoveryEventChain | None:
    """Select a deterministic failure -> change -> successful-test chain.

    Test failures are preferred over eligible Python command failures, while
    event order remains the tie-breaker within each failure class.
    """
    evidence_by_action = mutation_evidence or {}
    failures: list[tuple[int, AgentEvent, FailureEpisode]] = []
    for index, event in enumerate(events):
        action = planned_actions.get(event.action_id)
        failure = detect_failure(event, action)
        if failure is not None:
            if (
                failure.failure_type is FailureType.COMMAND_FAILURE
                and not _is_recovery_eligible_command_failure(action)
            ):
                continue
            failures.append((index, event, failure))

    candidates: list[tuple[tuple[int, int, int, int], RecoveryEventChain]] = []
    for failure_index, failure_event, failure in failures:
        for change_index in range(failure_index + 1, len(events)):
            change_event = events[change_index]
            if not change_event.result.success:
                continue
            change_action = planned_actions.get(change_event.action_id)
            if change_action is None:
                continue
            action_evidence = evidence_by_action.get(change_action.id)
            if not has_concrete_change(
                change_action,
                mutation_evidence=action_evidence,
            ):
                continue
            found_verification = False
            for verification_index in range(change_index + 1, len(events)):
                verification_event = events[verification_index]
                verification_action = planned_actions.get(verification_event.action_id)
                if (
                    not verification_event.result.success
                    or verification_event.result.exit_code != 0
                    or verification_action is None
                    or verification_action.tool != verification_event.result.tool_name
                    or not is_test_execution(verification_action)
                ):
                    continue
                is_test_failure = failure.failure_type.value == "test_failure"
                preference = 0 if is_test_failure else 1
                failure_order = failure_index if is_test_failure else -failure_index
                candidates.append(
                    (
                        (preference, failure_order, change_index, verification_index),
                        RecoveryEventChain(
                            failure_event=failure_event,
                            failure=failure,
                            change_event=change_event,
                            change_action=change_action,
                            verification_event=verification_event,
                        ),
                    )
                )
                found_verification = True
                break
            if found_verification:
                break
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _is_recovery_eligible_command_failure(action: PlannedAction | None) -> bool:
    """Recognize trusted Python execution as eligible command failure evidence."""
    if action is None or action.tool != "run_command":
        return False
    command = action.arguments.get("command")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        return False
    command_values = tuple(cast(Sequence[object], command))
    if not command_values or not all(isinstance(item, str) for item in command_values):
        return False
    argv = cast(tuple[str, ...], command_values)
    executable = argv[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if not (executable.startswith("python") or executable in {"py", "py.exe"}):
        return False
    if len(argv) < 2:
        return False
    if argv[1] in {"-c", "-m"}:
        return len(argv) >= 3 and bool(argv[2].strip())
    return not argv[1].startswith("-")


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

    failure = detect_failure(event, planned_action=action)
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
    mutation_evidence: Mapping[str, RepositoryMutationEvidence] | None = None,
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
    for event in event_list:
        persist_agent_event(
            repository,
            event,
            task,
            run,
            environment,
            planned_action=action_context.get(event.action_id),
        )
    chain = select_recovery_event_chain(
        event_list,
        action_context,
        mutation_evidence,
    )
    if chain is None:
        return None
    detected_failure = chain.failure
    change_action = chain.change_action
    successful_test_event = chain.verification_event
    normalized_change = normalize_planned_action(change_action)
    change_mutation_evidence = (mutation_evidence or {}).get(change_action.id)
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
        repository_mutation_evidence=change_mutation_evidence,
    )
    mutation_description = (
        "repository_mutation_before="
        f"{change_mutation_evidence.before_fingerprint}; "
        "repository_mutation_after="
        f"{change_mutation_evidence.after_fingerprint}; "
        if change_mutation_evidence is not None
        else ""
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
        f"{mutation_description}"
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


def persist_objective_anchored_recovery(
    repository: OperationalMemoryRepository,
    events: Sequence[AgentEvent],
    task: Task,
    run: Run,
    environment: EnvironmentContext,
    *,
    pre_agent_objective: ObjectiveAnchor,
    post_mutation_objective: ObjectiveAnchor,
    planned_actions: PlannedActionContext,
    mutation_evidence: Mapping[str, RepositoryMutationEvidence],
    objective_success_trigger_action_id: str,
) -> tuple[FailureEpisode, Resolution, Outcome]:
    """Persist R12's objective-anchored, mutation-backed recovery lineage.

    Objective anchors are audit observations, not agent events.  They are
    represented in the graph by deterministic synthetic pytest actions while
    the observed change relationship remains attached to the real mutating
    agent action.
    """
    if (
        pre_agent_objective.passed
        or pre_agent_objective.status != "test_failure"
        or pre_agent_objective.return_code in (None, 0)
    ):
        raise ValueError("R12 pre-agent objective anchor must represent failure")
    if (
        not post_mutation_objective.passed
        or post_mutation_objective.status != "passed"
        or post_mutation_objective.return_code != 0
    ):
        raise ValueError("R12 post-mutation objective anchor must represent success")
    _require_anchor_identity(pre_agent_objective, task, run)
    _require_anchor_identity(post_mutation_objective, task, run)

    action_context = _index_planned_actions(planned_actions)
    recovery_action = action_context.get(objective_success_trigger_action_id)
    if recovery_action is None:
        raise ValueError("R12 recovery action must exist in trusted planned_actions")
    if recovery_action.task_id != task.id or recovery_action.run_id != run.id:
        raise ValueError("R12 recovery action does not belong to this task/run")
    recovery_evidence = mutation_evidence.get(objective_success_trigger_action_id)
    if recovery_evidence is None or recovery_evidence.action_id != recovery_action.id:
        raise ValueError("R12 recovery action must have mutation evidence")
    recovery_event = next(
        (event for event in events if event.action_id == objective_success_trigger_action_id),
        None,
    )
    if recovery_event is None:
        raise ValueError("R12 recovery action must have a corresponding AgentEvent")
    if not recovery_event.result.success:
        raise ValueError("R12 recovery action event must be successful")
    _require_matching_identity(recovery_event, task, run)
    if not (
        pre_agent_objective.completed_at
        <= recovery_event.occurred_at
        <= post_mutation_objective.started_at
    ):
        raise ValueError("R12 objective and mutation timestamps are out of order")

    for event in events:
        planned = action_context.get(event.action_id)
        if planned is None:
            raise MissingTrustedPlannedActionError((event.action_id,))
        persist_agent_event(
            repository,
            event,
            task,
            run,
            environment,
            planned_action=planned,
        )

    failure_action, failure_event = _synthetic_objective_pair(
        run=run,
        task=task,
        anchor=pre_agent_objective,
        kind="failure",
        success=False,
        exit_code=pre_agent_objective.return_code,
    )
    detected_failure = persist_agent_event(
        repository,
        failure_event,
        task,
        run,
        environment,
        planned_action=failure_action,
    )
    if detected_failure is None:  # pragma: no cover - guarded by typed inputs
        raise ValueError("R12 synthetic objective failure was not detected")

    success_action, success_event = _synthetic_objective_pair(
        run=run,
        task=task,
        anchor=post_mutation_objective,
        kind="success",
        success=True,
        exit_code=0,
    )
    persist_agent_event(
        repository,
        success_event,
        task,
        run,
        environment,
        planned_action=success_action,
    )

    resolution_id = str(
        uuid5(NAMESPACE_URL, f"graph-swarm/resolution/{detected_failure.id}")
    )
    outcome_id = str(
        uuid5(NAMESPACE_URL, f"graph-swarm/outcome/{success_event.event_id}")
    )
    normalized_change = normalize_planned_action(recovery_action)
    evidence = ConcreteRecoveryEvidence(
        recovery_action=normalized_change,
        source_failure_id=detected_failure.id,
        resolution_id=resolution_id,
        objective_outcome_id=outcome_id,
        task_id=task.id,
        source_chronological_index=task.chronological_index,
        environment_id=environment.id,
        repository_mutation_evidence=recovery_evidence,
    )
    description = (
        "Observed objective-anchored recovery action: "
        "evidence_source=objective_anchored_v1; "
        f"action_id={evidence.recovery_action.id}; "
        f"run_id={run.id}; task_id={task.id}; repository={task.repository}; "
        f"repository_mutation_before={recovery_evidence.before_fingerprint}; "
        f"repository_mutation_after={recovery_evidence.after_fingerprint}; "
        f"source_failure_id={detected_failure.id}; "
        f"objective_success_action_id={success_action.id}."
    )
    resolution = Resolution(
        id=resolution_id,
        failure_id=detected_failure.id,
        description=description,
        status=ResolutionStatus.OBSERVED_SUCCESSFUL,
        successful_observations=1,
        failed_observations=0,
        observed_at=post_mutation_objective.completed_at,
    )
    outcome = Outcome(
        id=outcome_id,
        action_id=success_action.id,
        success=True,
        exit_code=0,
        observed_at=post_mutation_objective.completed_at,
    )
    repository.save_resolution(resolution)
    repository.link_failure_resolution(detected_failure.id, resolution.id)
    repository.save_outcome(outcome)
    repository.link_resolution_outcome(resolution.id, outcome.id)
    repository.link_resolution_observed_change(
        resolution.id,
        objective_success_trigger_action_id,
    )
    return detected_failure, resolution, outcome


def _require_anchor_identity(anchor: ObjectiveAnchor, task: Task, run: Run) -> None:
    if anchor.task_id != task.id or anchor.run_id != run.id:
        raise ValueError("objective anchor does not belong to this task/run")


def _synthetic_objective_pair(
    *,
    run: Run,
    task: Task,
    anchor: ObjectiveAnchor,
    kind: str,
    success: bool,
    exit_code: int | None,
) -> tuple[PlannedAction, AgentEvent]:
    action_id = str(
        uuid5(NAMESPACE_URL, f"graph-swarm/r12/{run.id}/objective-{kind}-action")
    )
    event_id = str(
        uuid5(NAMESPACE_URL, f"graph-swarm/r12/{run.id}/objective-{kind}-event")
    )
    action = PlannedAction(
        id=action_id,
        run_id=run.id,
        task_id=task.id,
        tool="run_command",
        operation="run_command",
        arguments={"command": ["python", "-m", "pytest"]},
        planned_at=anchor.started_at,
    )
    result = ActionResult(
        action_id=action_id,
        tool_name="run_command",
        success=success,
        exit_code=exit_code,
        output=(
            None
            if success
            else "frozen objective failed before agent execution"
        ),
        started_at=anchor.started_at,
        completed_at=anchor.completed_at,
    )
    event = AgentEvent(
        event_id=event_id,
        run_id=run.id,
        task_id=task.id,
        action_id=action_id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=result,
        occurred_at=anchor.completed_at,
    )
    return action, event


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
