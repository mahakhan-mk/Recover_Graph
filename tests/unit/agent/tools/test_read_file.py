from pathlib import Path

import pytest

from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.agent.tools.read_file import (
    MAX_READ_FILE_CHARS,
    MAX_READ_FILE_LENGTH,
    read_file,
)
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


def test_path_only_read_uses_default_page_without_metadata_for_short_file(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    (dependencies.workspace_root / "source.py").write_text("one\ntwo\n", encoding="utf-8")

    result = read_file(dependencies, "source.py")

    assert result.success is True
    assert result.output == "one\ntwo\n"


def test_path_only_read_is_bounded_by_default_length(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    source = "".join(f"line-{number}\n" for number in range(250))
    (dependencies.workspace_root / "source.py").write_text(source, encoding="utf-8")

    result = read_file(dependencies, "source.py")

    assert result.success is True
    assert result.output is not None
    assert "returned_lines=1-200" in result.output
    assert "total_lines=250" in result.output
    assert "next_offset=200" in result.output
    assert "line-199\n" in result.output
    assert "line-200\n" not in result.output


def test_offset_only_read_returns_continuation_metadata(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    (dependencies.workspace_root / "source.py").write_text(
        "one\ntwo\nthree\nfour\n", encoding="utf-8"
    )

    result = read_file(dependencies, "source.py", offset=1)

    assert result.success is True
    assert result.output is not None
    assert "returned_lines=2-4" in result.output
    assert "total_lines=4" in result.output
    assert "next_offset=none" in result.output
    assert result.output.endswith("two\nthree\nfour\n")


def test_offset_and_length_read_only_returns_requested_lines(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    (dependencies.workspace_root / "source.py").write_text(
        "one\ntwo\nthree\nfour\n", encoding="utf-8"
    )

    result = read_file(dependencies, "source.py", offset=1, length=2)

    assert result.success is True
    assert result.output is not None
    assert "returned_lines=2-3" in result.output
    assert "next_offset=3" in result.output
    assert result.output.endswith("two\nthree\n")


def test_offset_beyond_eof_returns_empty_page_metadata(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    (dependencies.workspace_root / "source.py").write_text("one\ntwo\n", encoding="utf-8")

    result = read_file(dependencies, "source.py", offset=20, length=2)

    assert result.success is True
    assert result.output == (
        "[read_file metadata: returned_lines=none; total_lines=2; next_offset=none]\n"
    )


@pytest.mark.parametrize(
    ("offset", "length", "message"),
    [
        (-1, 2, "offset must be greater than or equal to zero"),
        (0, 0, "length must be greater than zero"),
        (0, -1, "length must be greater than zero"),
        (0, MAX_READ_FILE_LENGTH + 1, f"length must not exceed {MAX_READ_FILE_LENGTH} lines"),
    ],
)
def test_invalid_page_arguments_return_failed_action_result(
    tmp_path: Path,
    offset: int,
    length: int,
    message: str,
) -> None:
    dependencies = make_dependencies(tmp_path)
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    (dependencies.workspace_root / "source.py").write_text("one\n", encoding="utf-8")

    result = read_file(dependencies, "source.py", offset=offset, length=length)

    assert result.success is False
    assert result.error == message
    assert len(dependencies.events) == 1


def test_page_output_character_bound_is_enforced(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    (dependencies.workspace_root / "source.py").write_text(
        "x" * (MAX_READ_FILE_CHARS + 1), encoding="utf-8"
    )

    result = read_file(dependencies, "source.py", length=1)

    assert result.success is False
    assert result.error == (
        f"requested content exceeds the {MAX_READ_FILE_CHARS}-character output bound"
    )


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
