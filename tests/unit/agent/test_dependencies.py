from datetime import UTC, datetime
from pathlib import Path

import pytest

from graph_swarm.agent.dependencies import (
    AgentDependencies,
    ExecutionRuntime,
    WorkspacePathError,
)
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


def make_docker_dependencies(tmp_path: Path) -> AgentDependencies:
    workspace = tmp_path / "workspace"
    return AgentDependencies(
        workspace,
        "run-001",
        "task-001",
        execution_runtime=ExecutionRuntime(
            runtime_type="docker",
            docker_executable=Path("docker.exe"),
            container_image="prepared:image",
            container_python_executable="/usr/bin/python3.12",
        ),
    )


def test_docker_container_workspace_path_resolves_to_host_workspace(
    tmp_path: Path,
) -> None:
    dependencies = make_docker_dependencies(tmp_path)

    assert dependencies.resolve_workspace_path("/workspace/src/example.py") == (
        dependencies.workspace_root / "src" / "example.py"
    ).resolve()


def test_docker_relative_workspace_path_still_resolves(tmp_path: Path) -> None:
    dependencies = make_docker_dependencies(tmp_path)

    assert dependencies.resolve_workspace_path("src/example.py") == (
        dependencies.workspace_root / "src" / "example.py"
    ).resolve()


def test_docker_container_workspace_root_resolves_to_workspace_root(tmp_path: Path) -> None:
    dependencies = make_docker_dependencies(tmp_path)

    assert dependencies.resolve_workspace_path("/workspace") == dependencies.workspace_root


@pytest.mark.parametrize("path", ["/etc/passwd", r"C:\outside.txt"])
def test_docker_host_absolute_paths_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(WorkspacePathError, match="absolute path|outside workspace"):
        make_docker_dependencies(tmp_path).resolve_workspace_path(path)


def test_docker_container_workspace_traversal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkspacePathError, match="outside workspace"):
        make_docker_dependencies(tmp_path).resolve_workspace_path("/workspace/../outside")


def test_local_runtime_does_not_accept_container_workspace_namespace(tmp_path: Path) -> None:
    dependencies = AgentDependencies(tmp_path / "workspace", "run-001", "task-001")

    with pytest.raises(WorkspacePathError, match="outside workspace"):
        dependencies.resolve_workspace_path("/workspace/src/example.py")


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


def test_docker_runtime_separates_host_cli_from_container_python() -> None:
    runtime = ExecutionRuntime(
        runtime_type="docker",
        docker_executable=Path("docker.exe"),
        container_image="prepared:image",
        container_python_executable="/opt/miniconda3/bin/python",
    )

    assert runtime.docker_executable == Path("docker.exe")
    assert runtime.container_python_executable == "/opt/miniconda3/bin/python"


def test_docker_runtime_requires_container_execution_metadata() -> None:
    with pytest.raises(ValueError, match="container_image"):
        ExecutionRuntime(
            runtime_type="docker",
            docker_executable=Path("docker.exe"),
            container_python_executable="/usr/bin/python3.10",
        )


def test_docker_runtime_rejects_a_host_path_as_python() -> None:
    with pytest.raises(ValueError, match="must not define python_executable"):
        ExecutionRuntime(
            runtime_type="docker",
            python_executable=Path("docker.exe"),
            docker_executable=Path("docker.exe"),
            container_image="prepared:image",
            container_python_executable="/usr/bin/python3.10",
        )
