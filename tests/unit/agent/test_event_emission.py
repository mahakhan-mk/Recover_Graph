import sys
from datetime import UTC, datetime
from pathlib import Path

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.agent.tools.read_file import read_file
from graph_swarm.agent.tools.run_command import run_command
from graph_swarm.agent.tools.write_file import write_file
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.events import AgentEventType


def make_dependencies(tmp_path: Path) -> AgentDependencies:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentDependencies(workspace, "run-001", "task-001")


def make_result(action_id: str) -> ActionResult:
    timestamp = datetime.now(UTC)
    return ActionResult(
        action_id=action_id,
        tool_name="read_file",
        success=True,
        output="contents",
        started_at=timestamp,
        completed_at=timestamp,
    )


def test_emitter_converts_and_collects_canonical_result(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    result = make_result("action-001")

    assert dependencies.events == []
    event = emit_action_event(dependencies, result)

    assert event.run_id == dependencies.run_id
    assert event.task_id == dependencies.task_id
    assert event.action_id == result.action_id
    assert event.result == result
    assert len(dependencies.events) == 1
    assert dependencies.events[0] == event
    assert event.event_type is AgentEventType.ACTION_COMPLETED
    assert event.event_id
    assert event.occurred_at.tzinfo is not None
    assert event.occurred_at.utcoffset() is not None


def test_emitted_events_have_unique_ids_and_serialize(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    first = emit_action_event(dependencies, make_result("action-001"))
    second = emit_action_event(dependencies, make_result("action-002"))

    assert first.event_id != second.event_id
    restored = type(first).model_validate_json(first.model_dump_json())
    assert restored == first


def test_sequential_tool_actions_accumulate_in_order(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    (dependencies.workspace_root / "input.txt").write_text("hello", encoding="utf-8")

    read_result = read_file(dependencies, "input.txt")
    write_result = write_file(dependencies, "output.txt", read_result.output or "")
    command_result = run_command(
        dependencies,
        [sys.executable, "-c", "print('done')"],
        timeout_seconds=5,
    )

    assert len(dependencies.events) == 3
    assert [event.result for event in dependencies.events] == [
        read_result,
        write_result,
        command_result,
    ]
    assert all(event.run_id == dependencies.run_id for event in dependencies.events)
    assert all(event.task_id == dependencies.task_id for event in dependencies.events)
    assert len({event.action_id for event in dependencies.events}) == 3
    assert len({event.event_id for event in dependencies.events}) == 3
