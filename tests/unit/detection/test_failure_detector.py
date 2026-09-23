from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from graph_swarm.detection.failure_detector import FailureDetector, detect_failure
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.domain.failures import FailureEpisode, FailureType

OCCURRED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_event(
    *,
    event_id: str = "event-001",
    action_id: str = "action-001",
    tool_name: str = "run_tests",
    success: bool = False,
    exit_code: int | None = 1,
    output: str | None = "pytest output",
    error: str | None = "one test failed\n",
) -> AgentEvent:
    result = ActionResult(
        action_id=action_id,
        tool_name=tool_name,
        success=success,
        exit_code=exit_code,
        output=output,
        error=error,
        started_at=OCCURRED_AT,
        completed_at=OCCURRED_AT,
    )
    return AgentEvent(
        event_id=event_id,
        run_id="run-001",
        task_id="task-001",
        action_id=action_id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=result,
        occurred_at=OCCURRED_AT,
    )


def test_failed_run_tests_creates_one_canonical_failure() -> None:
    event = make_event()

    failure = detect_failure(event)

    assert type(failure) is FailureEpisode
    assert failure is not None
    assert failure.id != event.event_id
    assert failure.id == str(uuid5(NAMESPACE_URL, event.event_id))
    assert UUID(failure.id).version == 5
    assert failure.action_id == event.action_id
    assert failure.failure_type is FailureType.TEST_FAILURE
    assert failure.observed_at == event.occurred_at
    assert failure.symptom == "one test failed"
    assert failure.signature == "run_tests:exit_code=1"


def test_passing_run_tests_creates_no_failure() -> None:
    event = make_event(success=True, exit_code=0, error=None)

    assert detect_failure(event) is None


def test_successful_unrelated_tool_creates_no_failure() -> None:
    event = make_event(tool_name="read_file", success=True, exit_code=None, error=None)

    assert detect_failure(event) is None


def test_failed_generic_run_command_is_a_command_failure() -> None:
    event = make_event(tool_name="run_command", success=False, exit_code=1)

    failure = FailureDetector().detect(event)
    assert failure is not None
    assert failure.failure_type is FailureType.COMMAND_FAILURE


def test_failed_pytest_run_command_is_a_test_failure_with_trusted_action() -> None:
    event = make_event(tool_name="run_command", success=False, exit_code=1)
    action = PlannedAction(
        id=event.action_id,
        run_id=event.run_id,
        task_id=event.task_id,
        tool="run_command",
        operation="run_command",
        arguments={"command": ["python", "-m", "pytest", "tests/test_locales.py", "-q"]},
        planned_at=event.result.started_at,
    )

    failure = FailureDetector().detect(event, action)

    assert failure is not None
    assert failure.failure_type is FailureType.TEST_FAILURE


def test_successful_generic_run_command_creates_no_failure() -> None:
    event = make_event(tool_name="run_command", success=True, exit_code=0, error=None)

    assert FailureDetector().detect(event) is None


def test_canonicalization_validation_failure_is_not_a_failure_episode() -> None:
    event = make_event(tool_name="run_command", success=False, exit_code=None)

    assert FailureDetector().detect(event) is None


def test_detection_is_deterministic_for_equivalent_events() -> None:
    first = detect_failure(make_event())
    second = detect_failure(make_event())

    assert first is not None
    assert second is not None
    assert first == second
