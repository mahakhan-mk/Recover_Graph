from pathlib import Path

import pytest

from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.agent.tools.edit_file import edit_file


def make_dependencies(tmp_path: Path) -> AgentDependencies:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentDependencies(workspace, "run-001", "task-001")


def test_exact_single_replacement_emits_one_event(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    target = dependencies.workspace_root / "notes.txt"
    target.write_text("alpha\nbroken\nomega\n", encoding="utf-8")

    result = edit_file(dependencies, "notes.txt", "broken", "fixed")

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "alpha\nfixed\nomega\n"
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result.action_id == result.action_id


def test_occurrence_mismatch_fails_without_modification(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    target = dependencies.workspace_root / "notes.txt"
    original = "alpha\nbroken\nomega\n"
    target.write_text(original, encoding="utf-8")

    result = edit_file(dependencies, "notes.txt", "missing", "fixed")

    assert result.success is False
    assert result.error == "exact edit expected 1 occurrence but found 0"
    assert target.read_text(encoding="utf-8") == original


def test_multiple_matches_require_explicit_expected_count(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    target = dependencies.workspace_root / "notes.txt"
    target.write_text("broken\nbroken\n", encoding="utf-8")

    rejected = edit_file(dependencies, "notes.txt", "broken", "fixed")
    accepted = edit_file(
        dependencies,
        "notes.txt",
        "broken",
        "repaired",
        expected_replacements=2,
    )

    assert rejected.success is False
    assert "found 2" in (rejected.error or "")
    assert accepted.success is True
    assert target.read_text(encoding="utf-8") == "repaired\nrepaired\n"


def test_empty_old_text_and_missing_file_fail(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    empty = edit_file(dependencies, "missing.txt", "", "fixed")
    missing = edit_file(dependencies, "missing.txt", "old", "fixed")

    assert empty.success is False
    assert empty.error == "old_text must not be empty"
    assert missing.success is False
    assert missing.error == "file does not exist"


def test_outside_workspace_and_directory_are_rejected(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (dependencies.workspace_root / "directory").mkdir()

    outside_result = edit_file(dependencies, str(outside), "secret", "changed")
    directory_result = edit_file(dependencies, "directory", "old", "new")

    assert outside_result.success is False
    assert outside_result.error == "path is outside the workspace"
    assert directory_result.success is False
    assert directory_result.error == "path is not a regular file"


def test_invalid_expected_replacements_fail_without_modification(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    target = dependencies.workspace_root / "notes.txt"
    target.write_text("old", encoding="utf-8")

    result = edit_file(
        dependencies,
        "notes.txt",
        "old",
        "new",
        expected_replacements=0,
    )

    assert result.success is False
    assert result.error == "expected_replacements must be at least 1"
    assert target.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("runtime_path", ["/workspace/../outside", "/etc/passwd"])
def test_docker_path_boundary_matches_other_controlled_tools(
    tmp_path: Path,
    runtime_path: str,
) -> None:
    dependencies = AgentDependencies(
        tmp_path / "workspace",
        "run-001",
        "task-001",
        execution_runtime=ExecutionRuntime(
            runtime_type="docker",
            docker_executable=Path("docker"),
            container_image="benchmark:image",
            container_python_executable="python",
        ),
    )
    dependencies.workspace_root.mkdir()
    (dependencies.workspace_root / "notes.txt").write_text("old", encoding="utf-8")

    result = edit_file(dependencies, runtime_path, "old", "new")

    assert result.success is False
    assert result.error == "path is outside the workspace"
