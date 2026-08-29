from pathlib import Path

import pytest

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.write_file import write_file
from graph_swarm.domain.action import ActionResult


def make_dependencies(tmp_path: Path) -> AgentDependencies:
    dependencies = AgentDependencies(tmp_path / "workspace", "run-001", "task-001")
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    return dependencies


def test_writes_text_file_inside_workspace(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)

    result = write_file(dependencies, "notes.txt", "hello workspace")

    assert type(result) is ActionResult
    assert result.tool_name == "write_file"
    assert result.success is True
    assert result.exit_code is None
    assert result.error is None
    assert result.output == "Wrote 15 characters to notes.txt"
    assert (dependencies.workspace_root / "notes.txt").read_text(encoding="utf-8") == (
        "hello workspace"
    )
    assert result.started_at.tzinfo is not None
    assert result.started_at.utcoffset() is not None
    assert result.completed_at.tzinfo is not None
    assert result.completed_at >= result.started_at
    assert result.action_id
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result == result
    assert dependencies.events[0].action_id == result.action_id


def test_repeated_writes_receive_different_action_ids(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)

    first = write_file(dependencies, "notes.txt", "first")
    second = write_file(dependencies, "other.txt", "second")

    assert first.action_id != second.action_id


def test_existing_file_can_be_overwritten(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    dependencies.workspace_root.mkdir(parents=True, exist_ok=True)
    target = dependencies.workspace_root / "notes.txt"
    target.write_text("old", encoding="utf-8")

    result = write_file(dependencies, "notes.txt", "new content")

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "new content"


@pytest.mark.parametrize("path", ["../outside.txt", ".."])
def test_workspace_escape_returns_failed_result(tmp_path: Path, path: str) -> None:
    result = write_file(make_dependencies(tmp_path), path, "blocked")

    assert result.success is False
    assert result.error == "path is outside the workspace"


def test_absolute_outside_path_returns_failed_result(tmp_path: Path) -> None:
    outside_path = tmp_path / "outside.txt"

    result = write_file(make_dependencies(tmp_path), str(outside_path), "blocked")

    assert result.success is False
    assert result.error == "path is outside the workspace"


def test_missing_parent_directory_fails_without_creating_directories(
    tmp_path: Path,
) -> None:
    dependencies = make_dependencies(tmp_path)

    result = write_file(dependencies, "missing/notes.txt", "content")

    assert result.success is False
    assert result.error == "parent directory does not exist"
    assert not (dependencies.workspace_root / "missing").exists()


def test_symlink_escape_is_blocked_when_supported(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = dependencies.workspace_root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    result = write_file(dependencies, "linked/secret.txt", "secret")

    assert result.success is False
    assert result.error == "path is outside the workspace"
