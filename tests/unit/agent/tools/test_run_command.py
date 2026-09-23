import subprocess
import sys
from pathlib import Path

import pytest

from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.agent.tools.run_command import (
    canonicalize_command,
    run_command,
)
from graph_swarm.domain.action import ActionResult


def make_dependencies(tmp_path: Path) -> AgentDependencies:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentDependencies(workspace, "run-001", "task-001")


def make_docker_dependencies(tmp_path: Path) -> AgentDependencies:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentDependencies(
        workspace,
        "run-001",
        "task-001",
        execution_runtime=ExecutionRuntime(
            runtime_type="docker",
            docker_executable=Path("docker.exe"),
            container_image="prepared:image",
            container_python_executable="/opt/miniconda3/bin/python",
        ),
    )


def assert_timestamps_are_valid(result: ActionResult) -> None:
    assert result.started_at.tzinfo is not None
    assert result.started_at.utcoffset() is not None
    assert result.completed_at.tzinfo is not None
    assert result.completed_at.utcoffset() is not None
    assert result.completed_at >= result.started_at


def test_single_string_command_is_canonicalized_to_posix_argv() -> None:
    assert canonicalize_command(["git status --short"]) == ["git", "status", "--short"]


def test_quoted_pytest_selector_survives_canonicalization() -> None:
    assert canonicalize_command(
        ["python -m pytest 'tests/test_locales.py::TestIcelandicLocale::test_format_timeframe' -q"]
    ) == [
        "python",
        "-m",
        "pytest",
        "tests/test_locales.py::TestIcelandicLocale::test_format_timeframe",
        "-q",
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["bash", "-lc", "echo hi | head"],
        ["sh", "-c", "echo hi"],
        ["/bin/bash", "-lc", "echo hi | head"],
        ["bash -lc 'echo hi | head'"],
    ],
)
def test_shell_launchers_are_rejected_fail_soft_before_execution(
    tmp_path: Path,
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_run(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("direct shell launcher was executed")

    monkeypatch.setattr("graph_swarm.agent.tools.run_command.subprocess.run", fake_run)
    dependencies = make_dependencies(tmp_path)
    result = run_command(dependencies, command, timeout_seconds=5)

    assert result.success is False
    assert result.exit_code is None
    assert result.error == (
        "run_command does not support direct shell invocation; "
        "invoke one executable with separate argv elements."
    )
    assert calls == []
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result == result


@pytest.mark.parametrize(
    "operator", ["|", "||", "&&", ";", ">", ">>", "<", "<<", "`", "$("]
)
def test_shell_operators_are_rejected_before_execution(
    tmp_path: Path,
    operator: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_run(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("shell-control command was executed")

    monkeypatch.setattr("graph_swarm.agent.tools.run_command.subprocess.run", fake_run)
    result = run_command(
        make_dependencies(tmp_path),
        [f"git status --short {operator} cat"],
        timeout_seconds=5,
    )

    assert result.success is False
    assert result.exit_code is None
    assert result.error == (
        "run_command requires structured argv; shell pipelines/redirection are unsupported. "
        "Invoke one executable with separate argv elements."
    )
    assert calls == []
    assert result.tool_name == "run_command"


def test_rejected_command_emits_failed_event_and_valid_command_can_follow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = make_dependencies(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "valid command ran\n", "")

    monkeypatch.setattr("graph_swarm.agent.tools.run_command.subprocess.run", fake_run)

    rejected = run_command(
        dependencies,
        ["python -c \"print('rejected')\" | cat"],
        timeout_seconds=5,
    )
    valid = run_command(
        dependencies,
        ["python -c \"print('valid')\""],
        timeout_seconds=5,
    )

    assert rejected.success is False
    assert valid.success is True
    assert calls == [["python", "-c", "print('valid')"]]
    assert [event.result for event in dependencies.events] == [rejected, valid]


def test_shell_metacharacters_inside_python_argument_remain_allowed(tmp_path: Path) -> None:
    assert canonicalize_command(["python", "-c", "print(1); print(2)"]) == [
        "python",
        "-c",
        "print(1); print(2)",
    ]

    result = run_command(
        make_dependencies(tmp_path),
        [sys.executable, "-c", "print(1); print(2)"],
        timeout_seconds=5,
    )

    assert result.success is True
    assert result.output is not None
    assert result.output.splitlines() == ["1", "2"]


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


def test_docker_command_wraps_arbitrary_argv_in_prepared_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = make_docker_dependencies(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "container output\n", "")

    monkeypatch.setattr("graph_swarm.agent.tools.run_command.subprocess.run", fake_run)

    result = run_command(
        dependencies,
        ["python", "-c", "print('inside')"],
        timeout_seconds=5,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:4] == ["docker.exe", "run", "--rm", "--network"]
    assert command[4:6] == ["none", "--mount"]
    assert command[6] == (
        f"type=bind,source={dependencies.workspace_root},target=/workspace"
    )
    assert command[7:11] == [
        "--workdir",
        "/workspace",
        "--env",
        "PYTHONPATH=/workspace/src:/workspace",
    ]
    assert command[11:] == ["prepared:image", "python", "-c", "print('inside')"]
    assert result.success is True
    assert result.output == "container output\n"
    assert dependencies.events[0].result == result


def test_docker_command_failure_preserves_tool_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = make_docker_dependencies(tmp_path)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 7, "partial output", "container error")

    monkeypatch.setattr("graph_swarm.agent.tools.run_command.subprocess.run", fake_run)

    result = run_command(dependencies, ["pytest", "-q"], timeout_seconds=5)

    assert result.success is False
    assert result.exit_code == 7
    assert result.output == "partial output"
    assert result.error == "container error"
    assert len(dependencies.events) == 1
    assert dependencies.events[0].result == result
