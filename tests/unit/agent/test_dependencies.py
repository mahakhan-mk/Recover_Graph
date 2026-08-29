from datetime import UTC, datetime
from pathlib import Path

import pytest

from graph_swarm.agent.dependencies import AgentDependencies, WorkspacePathError
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.events import AgentEvent, AgentEventType


def make_event() -> AgentEvent:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    result = ActionResult(
        action_id="action-001",
        tool_name="read_file",
        success=True,
        output="contents",
        started_at=timestamp,
        completed_at=timestamp,
    )
    return AgentEvent(
        event_id="event-001",
        run_id="run-001",
        task_id="task-001",
        action_id="action-001",
        event_type=AgentEventType.ACTION_COMPLETED,
        result=result,
        occurred_at=timestamp,
    )


def test_workspace_root_is_resolved_and_ids_are_retained(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    dependencies = AgentDependencies(workspace, "run-001", "task-001")

    assert dependencies.workspace_root == workspace.resolve()
    assert dependencies.workspace_root.is_absolute()
    assert dependencies.run_id == "run-001"
    assert dependencies.task_id == "task-001"


def test_event_collection_starts_empty_and_stores_canonical_event(tmp_path: Path) -> None:
    dependencies = AgentDependencies(tmp_path, "run-001", "task-001")
    event = make_event()

    assert dependencies.events == []
    dependencies.events.append(event)

    assert dependencies.events == [event]
    assert type(dependencies.events[0]) is AgentEvent


def test_valid_nested_workspace_path_resolves(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "src" / "module.py"
    dependencies = AgentDependencies(workspace, "run-001", "task-001")

    assert dependencies.resolve_workspace_path("src/module.py") == nested.resolve()


def test_parent_traversal_outside_workspace_is_rejected(tmp_path: Path) -> None:
    dependencies = AgentDependencies(tmp_path / "workspace", "run-001", "task-001")

    with pytest.raises(WorkspacePathError, match="outside workspace"):
        dependencies.resolve_workspace_path("../outside.txt")


def test_absolute_outside_workspace_path_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    dependencies = AgentDependencies(workspace, "run-001", "task-001")

    with pytest.raises(WorkspacePathError, match="outside workspace"):
        dependencies.resolve_workspace_path(outside)


def test_empty_run_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        AgentDependencies(tmp_path, "", "task-001")


def test_empty_task_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task_id"):
        AgentDependencies(tmp_path, "run-001", "")
