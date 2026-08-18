import inspect
from datetime import UTC, datetime, timedelta
from typing import get_type_hints

import pytest

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.events import AgentEvent
from graph_swarm.graph._validation import (
    validate_action_persistence,
    validate_relationship_ids,
)
from graph_swarm.graph.read_models import ActionLineageRecord, IncidentLineage
from graph_swarm.graph.repository import OperationalMemoryRepository

STARTED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
COMPLETED_AT = STARTED_AT + timedelta(seconds=1)

ENTITY_METHODS = (
    "save_run",
    "save_task",
    "save_action",
    "save_tool",
    "save_environment",
    "save_failure",
    "save_resolution",
    "save_outcome",
)
RELATIONSHIP_METHODS = (
    "link_task_action",
    "link_action_tool",
    "link_action_run",
    "link_action_failure",
    "link_failure_environment",
    "link_failure_resolution",
    "link_resolution_outcome",
)


def test_all_repository_methods_are_synchronous() -> None:
    for method_name in (*ENTITY_METHODS, *RELATIONSHIP_METHODS, "get_incident_lineage"):
        method = getattr(OperationalMemoryRepository, method_name)
        assert not inspect.iscoroutinefunction(method)


def test_entity_methods_return_none_and_use_typed_domain_models() -> None:
    expected_types = {
        "save_run": ("run", "Run"),
        "save_task": ("task", "Task"),
        "save_action": ("action", "PlannedAction"),
        "save_tool": ("tool", "Tool"),
        "save_environment": ("environment", "EnvironmentContext"),
        "save_failure": ("failure", "FailureEpisode"),
        "save_resolution": ("resolution", "Resolution"),
        "save_outcome": ("outcome", "Outcome"),
    }

    for method_name, (parameter_name, expected_type_name) in expected_types.items():
        method_hints = get_type_hints(getattr(OperationalMemoryRepository, method_name))
        assert method_hints[parameter_name].__name__ == expected_type_name
        assert method_hints["return"] is type(None)


def test_save_action_uses_frozen_action_contracts() -> None:
    hints = get_type_hints(OperationalMemoryRepository.save_action)

    assert hints["action"] is PlannedAction
    assert hints["result"] is ActionResult
    assert ActionResult.__module__ == "graph_swarm.domain.action"
    assert AgentEvent.__module__ == "graph_swarm.domain.events"


def make_planned_action(action_id: str = "action-001") -> PlannedAction:
    return PlannedAction(
        id=action_id,
        run_id="run-001",
        task_id="task-001",
        tool="run_tests",
        operation="pytest",
        planned_at=STARTED_AT,
    )


def make_action_result(action_id: str = "action-001") -> ActionResult:
    return ActionResult(
        action_id=action_id,
        tool_name="run_tests",
        success=True,
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
    )


def test_action_persistence_rejects_mismatched_action_ids() -> None:
    action = make_planned_action()
    result = make_action_result("action-002")

    with pytest.raises(ValueError, match="must match"):
        validate_action_persistence(action, result)

    with pytest.raises(ValueError, match="must match"):
        ActionLineageRecord(planned_action=action, result=result)


def test_relationship_identity_validation_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="task_id"):
        validate_relationship_ids(task_id="", action_id="action-001")


def test_incident_lineage_is_typed_not_a_dictionary() -> None:
    assert not issubclass(IncidentLineage, dict)
    assert not issubclass(ActionLineageRecord, dict)
    assert (
        get_type_hints(OperationalMemoryRepository.get_incident_lineage)["return"]
        is IncidentLineage
    )
