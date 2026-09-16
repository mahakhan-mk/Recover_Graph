from pathlib import Path

import pytest

from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.agent.tools.read_file import read_file
from graph_swarm.domain.action import ActionResult


def make_dependencies(tmp_path: Path) -> AgentDependencies:
    return AgentDependencies(tmp_path / "workspace", "run-001", "task-001")


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


def test_reads_text_file_inside_workspace(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    file_path = dependencies.workspace_root / "notes.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("hello workspace", encoding="utf-8")

    result = read_file(dependencies, "notes.txt")

    assert type(result) is ActionResult
    assert result.tool_name == "read_file"
    assert result.success is True
    assert result.exit_code is None
    assert result.output == "hello workspace"
    assert result.error is None
    assert result.started_at.tzinfo is not None
    assert result.started_at.utcoffset() is not None
    assert result.completed_at.tzinfo is not None
    assert result.completed_at >= result.started_at
    assert result.action_id
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result == result
    assert dependencies.events[0].action_id == result.action_id


def test_docker_read_accepts_container_workspace_path(tmp_path: Path) -> None:
    dependencies = make_docker_dependencies(tmp_path)
    file_path = dependencies.workspace_root / "src" / "example.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("print('example')", encoding="utf-8")

    result = read_file(dependencies, "/workspace/src/example.py")

    assert result.success is True
    assert result.output == "print('example')"


@pytest.mark.parametrize("path", ["/workspace/../outside", "/etc/passwd", r"C:\outside.txt"])
def test_docker_read_rejects_paths_outside_container_workspace(
    tmp_path: Path,
    path: str,
) -> None:
    result = read_file(make_docker_dependencies(tmp_path), path)

    assert result.success is False
    assert result.error == "path is outside the workspace"


def test_repeated_reads_receive_different_action_ids(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    (dependencies.workspace_root / "notes.txt").write_text("text", encoding="utf-8")

    first = read_file(dependencies, "notes.txt")
    second = read_file(dependencies, "notes.txt")

    assert first.action_id != second.action_id


def test_nonexistent_file_returns_failed_result(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    result = read_file(dependencies, "missing.txt")

    assert result.success is False
    assert result.exit_code is None
    assert result.output is None
    assert result.error == "file does not exist"
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result == result


@pytest.mark.parametrize("path", ["../outside.txt", ".."])
def test_workspace_escape_returns_failed_result(tmp_path: Path, path: str) -> None:
    result = read_file(make_dependencies(tmp_path), path)

    assert result.success is False
    assert result.error == "path is outside the workspace"


def test_absolute_outside_path_returns_failed_result(tmp_path: Path) -> None:
    outside_path = tmp_path / "outside.txt"

    result = read_file(make_dependencies(tmp_path), str(outside_path))

    assert result.success is False
    assert result.error == "path is outside the workspace"


def test_symlink_escape_is_blocked_when_supported(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = dependencies.workspace_root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    result = read_file(dependencies, "linked/secret.txt")

    assert result.success is False
    assert result.error == "path is outside the workspace"
