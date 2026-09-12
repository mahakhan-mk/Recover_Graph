from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.integration.event_persistence import persist_agent_event_stream
from graph_swarm.memory.recovery_evidence import (
    ConcreteRecoveryEvidence,
    has_concrete_change,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_context() -> tuple[Task, Run, EnvironmentContext]:
    task = Task(
        id="task-001",
        problem_statement="Fix the repository so its tests pass.",
        family_id="family-001",
        repository="example/repository",
        chronological_index=7,
    )
    return (
        task,
        Run(id="run-001", task_id=task.id, started_at=NOW),
        EnvironmentContext(
            id="environment-001",
            repository=task.repository,
            runtime="python-3.13",
        ),
    )


def make_action(
    action_id: str,
    tool: str,
    operation: str,
    arguments: dict[str, object],
    offset: int,
) -> PlannedAction:
    return PlannedAction(
        id=action_id,
        run_id="run-001",
        task_id="task-001",
        tool=tool,
        operation=operation,
        arguments=arguments,
        planned_at=NOW + timedelta(seconds=offset),
    )


def make_event(
    action: PlannedAction,
    *,
    event_id: str,
    success: bool,
    exit_code: int | None,
    output: str | None,
) -> AgentEvent:
    completed_at = action.planned_at + timedelta(milliseconds=100)
    return AgentEvent(
        event_id=event_id,
        run_id=action.run_id,
        task_id=action.task_id,
        action_id=action.id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=ActionResult(
            action_id=action.id,
            tool_name=action.tool,
            success=success,
            exit_code=exit_code,
            output=output,
            started_at=action.planned_at,
            completed_at=completed_at,
        ),
        occurred_at=completed_at,
    )


def make_recovery_stream() -> tuple[list[AgentEvent], tuple[PlannedAction, ...]]:
    failed_test = make_action("action-failed", "run_tests", "run_tests", {}, 0)
    read_action = make_action(
        "action-read",
        "read_file",
        "read_file",
        {"path": "calculator.py"},
        1,
    )
    write_action = make_action(
        "action-write",
        "write_file",
        "write_file",
        {
            "content": "def add(a, b):\n    return a + b\n",
            "path": "calculator.py",
        },
        2,
    )
    successful_test = make_action("action-success", "run_tests", "run_tests", {}, 3)
    events = [
        make_event(
            failed_test,
            event_id="event-failed",
            success=False,
            exit_code=1,
            output="1 failed",
        ),
        make_event(
            read_action,
            event_id="event-read",
            success=True,
            exit_code=None,
            output="old source",
        ),
        make_event(
            write_action,
            event_id="event-write",
            success=True,
            exit_code=None,
            output="Wrote 34 characters to calculator.py",
        ),
        make_event(
            successful_test,
            event_id="event-success",
            success=True,
            exit_code=0,
            output="all passed",
        ),
    ]
    return events, (failed_test, read_action, write_action, successful_test)


def test_concrete_change_and_objective_success_create_observed_lineage() -> None:
    task, run, environment = make_context()
    events, planned_actions = make_recovery_stream()
    repository = Mock(spec=OperationalMemoryRepository)

    result = persist_agent_event_stream(
        repository,
        events,
        task,
        run,
        environment,
        planned_actions=planned_actions,
    )

    assert result is not None
    failure, resolution, outcome = result
    assert resolution.failure_id == failure.id
    assert outcome.success is True
    assert outcome.action_id == "action-success"
    assert resolution.status.value == "observed_successful"
    assert "source_failure_id=" + failure.id in resolution.description
    assert "objective_success_action_id=action-success" in resolution.description
    repository.link_failure_resolution.assert_called_once_with(
        failure.id,
        resolution.id,
    )
    repository.link_resolution_outcome.assert_called_once_with(
        resolution.id,
        outcome.id,
    )
    repository.link_resolution_observed_change.assert_called_once_with(
        resolution.id,
        "action-write",
    )


def test_real_normalized_arguments_are_saved_and_not_reconstructed_empty() -> None:
    task, run, environment = make_context()
    events, planned_actions = make_recovery_stream()
    repository = Mock(spec=OperationalMemoryRepository)

    persist_agent_event_stream(
        repository,
        events,
        task,
        run,
        environment,
        planned_actions=planned_actions,
    )

    saved_actions = {
        call.args[0].id: call.args[0]
        for call in repository.save_action.call_args_list
    }
    assert saved_actions["action-write"].arguments == {
        "content": "def add(a, b):\n    return a + b\n",
        "path": "calculator.py",
    }
    assert saved_actions["action-write"].arguments != {}
    resolution = repository.save_resolution.call_args.args[0]
    assert '"content":"def add(a, b):\\n    return a + b\\n"' in resolution.description
    assert '"path":"calculator.py"' in resolution.description
    assert "Wrote 34 characters" not in resolution.description


def test_structured_command_arguments_are_preserved_deterministically() -> None:
    task, run, environment = make_context()
    failed_test = make_action("action-failed", "run_tests", "run_tests", {}, 0)
    command_action = make_action(
        "action-command",
        "run_command",
        "run_command",
        {"command": ("python", "-m", "pytest", "tests")},
        1,
    )
    successful_test = make_action("action-success", "run_tests", "run_tests", {}, 2)
    events = [
        make_event(
            failed_test,
            event_id="event-failed",
            success=False,
            exit_code=1,
            output="failed",
        ),
        make_event(
            command_action,
            event_id="event-command",
            success=True,
            exit_code=0,
            output="command complete",
        ),
        make_event(
            successful_test,
            event_id="event-success",
            success=True,
            exit_code=0,
            output="passed",
        ),
    ]
    repository = Mock(spec=OperationalMemoryRepository)

    persist_agent_event_stream(
        repository,
        events,
        task,
        run,
        environment,
        planned_actions=(failed_test, command_action, successful_test),
    )

    saved_command = next(
        call.args[0]
        for call in repository.save_action.call_args_list
        if call.args[0].id == command_action.id
    )
    assert saved_command.arguments == {
        "command": ["python", "-m", "pytest", "tests"]
    }


def test_read_and_test_actions_cannot_be_concrete_recovery_changes() -> None:
    read_action = make_action(
        "action-read",
        "read_file",
        "read_file",
        {"path": "calculator.py"},
        1,
    )
    test_action = make_action("action-test", "run_tests", "run_tests", {}, 1)

    assert has_concrete_change(read_action) is False
    assert has_concrete_change(test_action) is False

    try:
        ConcreteRecoveryEvidence(
            recovery_action=read_action,
            source_failure_id="failure-001",
            resolution_id="resolution-001",
            objective_outcome_id="outcome-001",
            task_id="task-001",
            source_chronological_index=7,
            environment_id="environment-001",
        )
    except ValueError as error:
        assert "concrete change" in str(error)
    else:
        raise AssertionError("read-only action was accepted as recovery evidence")


def test_write_action_requires_path_and_concrete_content() -> None:
    missing_content = make_action(
        "action-write",
        "write_file",
        "write_file",
        {"path": "calculator.py"},
        1,
    )
    valid_write = missing_content.model_copy(
        update={"arguments": {"path": "calculator.py", "content": "new source"}}
    )

    assert has_concrete_change(missing_content) is False
    assert has_concrete_change(valid_write) is True


def test_generic_success_without_identifiable_change_does_not_create_resolution() -> None:
    task, run, environment = make_context()
    failed_test = make_action("action-failed", "run_tests", "run_tests", {}, 0)
    generic_action = make_action("action-generic", "read_file", "read_file", {}, 1)
    successful_test = make_action("action-success", "run_tests", "run_tests", {}, 2)
    events = [
        make_event(
            failed_test,
            event_id="event-failed",
            success=False,
            exit_code=1,
            output="failed",
        ),
        make_event(
            generic_action,
            event_id="event-generic",
            success=True,
            exit_code=None,
            output="Wrote 123 characters to nowhere",
        ),
        make_event(
            successful_test,
            event_id="event-success",
            success=True,
            exit_code=0,
            output="passed",
        ),
    ]
    repository = Mock(spec=OperationalMemoryRepository)

    result = persist_agent_event_stream(
        repository,
        events,
        task,
        run,
        environment,
        planned_actions=(failed_test, generic_action, successful_test),
    )

    assert result is None
    repository.save_resolution.assert_not_called()
    repository.link_resolution_observed_change.assert_not_called()


def test_missing_planned_context_can_persist_events_but_not_recovery_claim() -> None:
    task, run, environment = make_context()
    events, _ = make_recovery_stream()
    repository = Mock(spec=OperationalMemoryRepository)

    result = persist_agent_event_stream(repository, events, task, run, environment)

    assert result is None
    saved_write = next(
        call.args[0]
        for call in repository.save_action.call_args_list
        if call.args[0].id == "action-write"
    )
    assert saved_write.arguments == {}
    repository.save_resolution.assert_not_called()


def test_repeated_persistence_reuses_stable_lineage_ids_and_relationship() -> None:
    task, run, environment = make_context()
    events, planned_actions = make_recovery_stream()
    repository = Mock(spec=OperationalMemoryRepository)

    first = persist_agent_event_stream(
        repository,
        events,
        task,
        run,
        environment,
        planned_actions=planned_actions,
    )
    second = persist_agent_event_stream(
        repository,
        events,
        task,
        run,
        environment,
        planned_actions=planned_actions,
    )

    assert first is not None
    assert second is not None
    assert tuple(item.id for item in first) == tuple(item.id for item in second)
    assert repository.link_resolution_observed_change.call_count == 2
