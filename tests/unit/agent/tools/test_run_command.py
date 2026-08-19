import sys
from pathlib import Path

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.run_command import run_command
from graph_swarm.domain.action import ActionResult


def make_dependencies(tmp_path: Path) -> AgentDependencies:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentDependencies(workspace, "run-001", "task-001")


def assert_timestamps_are_valid(result: ActionResult) -> None:
    assert result.started_at.tzinfo is not None
    assert result.started_at.utcoffset() is not None
    assert result.completed_at.tzinfo is not None
    assert result.completed_at.utcoffset() is not None
    assert result.completed_at >= result.started_at


def test_successful_command_returns_canonical_result(tmp_path: Path) -> None:
    result = run_command(
        make_dependencies(tmp_path),
        [sys.executable, "-c", "print('hello')"],
        timeout_seconds=5,
    )

    assert type(result) is ActionResult
    assert result.tool_name == "run_command"
    assert result.success is True
    assert result.exit_code == 0
    assert result.output is not None and "hello" in result.output
    assert result.action_id
    assert_timestamps_are_valid(result)


def test_nonzero_command_retains_stdout_and_exit_code(tmp_path: Path) -> None:
    result = run_command(
        make_dependencies(tmp_path),
        [sys.executable, "-c", "import sys; print('failed'); sys.exit(3)"],
        timeout_seconds=5,
    )

    assert result.success is False
    assert result.exit_code == 3
    assert result.output is not None and "failed" in result.output


def test_stderr_does_not_make_successful_command_fail(tmp_path: Path) -> None:
    result = run_command(
        make_dependencies(tmp_path),
        [sys.executable, "-c", "import sys; sys.stderr.write('warning')"],
        timeout_seconds=5,
    )

    assert result.success is True
    assert result.exit_code == 0
    assert result.error == "warning"


def test_command_runs_with_workspace_as_cwd(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    result = run_command(
        dependencies,
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        timeout_seconds=5,
    )

    assert result.success is True
    assert result.output is not None
    assert str(dependencies.workspace_root) in result.output


def test_timeout_returns_failed_result(tmp_path: Path) -> None:
    result = run_command(
        make_dependencies(tmp_path),
        [sys.executable, "-c", "import time; time.sleep(1)"],
        timeout_seconds=0.05,
    )

    assert result.success is False
    assert result.exit_code is None
    assert result.error is not None and "timed out" in result.error


def test_nonexistent_executable_returns_failed_result(tmp_path: Path) -> None:
    result = run_command(
        make_dependencies(tmp_path),
        ["graph-swarm-executable-that-does-not-exist"],
        timeout_seconds=5,
    )

    assert result.success is False
    assert result.exit_code is None
    assert result.error == "executable was not found"


def test_empty_command_returns_failed_result(tmp_path: Path) -> None:
    result = run_command(make_dependencies(tmp_path), [], timeout_seconds=5)

    assert result.success is False
    assert result.exit_code is None
    assert result.error == "command must not be empty"


def test_repeated_commands_receive_different_action_ids(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    command = [sys.executable, "-c", "print('ok')"]

    first = run_command(dependencies, command, timeout_seconds=5)
    second = run_command(dependencies, command, timeout_seconds=5)

    assert first.action_id != second.action_id
    assert len(dependencies.events) == 2
    assert dependencies.events[0].result == first
    assert dependencies.events[1].result == second


def test_nonzero_command_emits_one_matching_event(tmp_path: Path) -> None:
    dependencies = make_dependencies(tmp_path)
    result = run_command(
        dependencies,
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        timeout_seconds=5,
    )

    assert result.success is False
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result == result
    assert dependencies.events[0].action_id == result.action_id
