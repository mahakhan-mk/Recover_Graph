from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.events import AgentEvent, AgentEventType

STARTED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
COMPLETED_AT = STARTED_AT + timedelta(seconds=2)


def test_successful_action_result() -> None:
    result = ActionResult(
        action_id="action-001",
        tool_name="read_file",
        success=True,
        exit_code=None,
        output="file contents",
        error=None,
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
    )

    assert result.success is True
    assert result.output == "file contents"
    assert result.error is None


def test_failed_test_execution_action_result() -> None:
    result = ActionResult(
        action_id="action-002",
        tool_name="run_tests",
        success=False,
        exit_code=1,
        output="pytest output",
        error="diagnostic output",
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
    )

    assert result.success is False
    assert result.exit_code == 1
    assert result.output == "pytest output"
    assert result.error == "diagnostic output"


def test_successful_command_action_result() -> None:
    result = ActionResult(
        action_id="action-003",
        tool_name="run_command",
        success=True,
        exit_code=0,
        output="completed",
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
    )

    assert result.exit_code == 0
    assert result.success is True


def test_action_result_rejects_completed_before_started() -> None:
    with pytest.raises(ValidationError, match="completed_at"):
        ActionResult(
            action_id="action-004",
            tool_name="run_tests",
            success=False,
            started_at=COMPLETED_AT,
            completed_at=STARTED_AT,
        )


@pytest.mark.parametrize("timestamp_field", ["started_at", "completed_at"])
def test_action_result_rejects_naive_timestamps(timestamp_field: str) -> None:
    values = {
        "action_id": "action-005",
        "tool_name": "read_file",
        "success": True,
        "started_at": STARTED_AT,
        "completed_at": COMPLETED_AT,
    }
    values[timestamp_field] = datetime(2026, 1, 1, 12, 0)

    with pytest.raises(ValidationError, match="timezone-aware"):
        ActionResult.model_validate(values)


def make_event(action_id: str = "action-006") -> AgentEvent:
    result = ActionResult(
        action_id=action_id,
        tool_name="run_tests",
        success=False,
        exit_code=1,
        output="pytest output",
        error="one test failed",
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
    )
    return AgentEvent(
        event_id="event-001",
        run_id="run-001",
        task_id="GS-T001",
        action_id=action_id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=result,
        occurred_at=COMPLETED_AT,
    )


def test_agent_event_wraps_action_result() -> None:
    event = make_event()

    assert event.event_type is AgentEventType.ACTION_COMPLETED
    assert event.result.tool_name == "run_tests"
    assert event.action_id == event.result.action_id


def test_agent_event_rejects_mismatched_action_id() -> None:
    result = ActionResult(
        action_id="action-result",
        tool_name="read_file",
        success=True,
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
    )

    with pytest.raises(ValidationError, match="action_id"):
        AgentEvent(
            event_id="event-002",
            run_id="run-001",
            task_id="GS-T001",
            action_id="action-event",
            event_type=AgentEventType.ACTION_COMPLETED,
            result=result,
            occurred_at=COMPLETED_AT,
        )


def test_agent_event_rejects_naive_occurred_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        AgentEvent(
            event_id="event-003",
            run_id="run-001",
            task_id="GS-T001",
            action_id="action-006",
            event_type=AgentEventType.ACTION_COMPLETED,
            result=make_event().result,
            occurred_at=datetime(2026, 1, 1, 12, 0),
        )


def test_failed_event_json_round_trip_preserves_failure_fields() -> None:
    event = make_event()

    restored = AgentEvent.model_validate_json(event.model_dump_json())

    assert restored == event
    assert restored.event_id == "event-001"
    assert restored.result.tool_name == "run_tests"
    assert restored.result.success is False
    assert restored.result.exit_code == 1
    assert restored.result.output == "pytest output"
    assert restored.result.error == "one test failed"


def test_contract_classes_are_canonical_domain_models() -> None:
    assert ActionResult.__module__ == "graph_swarm.domain.action"
    assert AgentEvent.__module__ == "graph_swarm.domain.events"
