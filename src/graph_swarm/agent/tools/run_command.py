"""Controlled structured-argv process execution tool."""

import os
import shlex
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.domain.action import ActionResult

_SHELL_OPERATOR_TOKENS = frozenset({"|", "||", "&&", ";", ">", ">>", "<", "<<", "`"})
_UNSUPPORTED_SHELL_CONTROL_ERROR = (
    "run_command requires structured argv; shell pipelines/redirection are unsupported. "
    "Invoke one executable with separate argv elements."
)
_SHELL_EXECUTABLE_BASENAMES = frozenset({"sh", "bash", "dash", "ash", "zsh", "ksh"})
_UNSUPPORTED_SHELL_LAUNCHER_ERROR = (
    "run_command does not support direct shell invocation; "
    "invoke one executable with separate argv elements."
)


class CommandCanonicalizationError(ValueError):
    """Raised when model command input is not safe structured argv."""


def _shell_control_outside_quotes(value: str) -> bool:
    single = False
    double = False
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\" and not single:
            escaped = True
            continue
        if character == "'" and not double:
            single = not single
            continue
        if character == '"' and not single:
            double = not double
            continue
        if single or double:
            continue
        if character in "|;&><`" or value[index : index + 2] == "$(":
            return True
    return False


def _is_direct_shell_launcher(values: Sequence[str]) -> bool:
    executable = values[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if executable.endswith(".exe"):
        executable = executable[:-4]
    return executable in _SHELL_EXECUTABLE_BASENAMES


def canonicalize_command(command: Sequence[str]) -> list[str]:
    """Canonicalize model command input into deterministic POSIX argv.

    A one-element command is treated as a model-emitted command string and
    tokenized with POSIX ``shlex`` rules.  Already structured argv is retained
    verbatim, apart from rejecting standalone shell operators.
    """
    values = list(command)
    if not values:
        raise CommandCanonicalizationError("run_command requires non-empty structured argv")
    if any(not isinstance(value, str) for value in cast(Sequence[object], values)):
        raise CommandCanonicalizationError("run_command requires structured argv strings")
    if len(values) == 1:
        raw = values[0]
        if _shell_control_outside_quotes(raw):
            raise CommandCanonicalizationError(_UNSUPPORTED_SHELL_CONTROL_ERROR)
        try:
            values = shlex.split(raw, posix=True)
        except ValueError as error:
            raise CommandCanonicalizationError(
                "run_command requires valid POSIX structured argv"
            ) from error
    elif any(value in _SHELL_OPERATOR_TOKENS or "$(" in value for value in values):
        raise CommandCanonicalizationError(_UNSUPPORTED_SHELL_CONTROL_ERROR)
    if not values:
        raise CommandCanonicalizationError("run_command requires non-empty structured argv")
    if _is_direct_shell_launcher(values):
        raise CommandCanonicalizationError(_UNSUPPORTED_SHELL_LAUNCHER_ERROR)
    return values


def _completion_time(started_at: datetime) -> datetime:
    return max(datetime.now(UTC), started_at)


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _result(
    action_id: str,
    tool_name: str,
    started_at: datetime,
    success: bool,
    exit_code: int | None,
    output: str | None,
    error: str | None,
) -> ActionResult:
    return ActionResult(
        action_id=action_id,
        tool_name=tool_name,
        success=success,
        exit_code=exit_code,
        output=output,
        error=error,
        started_at=started_at,
        completed_at=_completion_time(started_at),
    )


def workspace_process_environment(workspace: Path) -> dict[str, str]:
    """Build a process environment rooted only in the selected workspace."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    import_paths = [workspace, workspace / "src"]
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in import_paths if path.exists())
    return environment


def _runtime_for_dependencies(dependencies: AgentDependencies) -> ExecutionRuntime:
    """Return the explicit runtime, or the historical local runtime."""
    if dependencies.execution_runtime is not None:
        return dependencies.execution_runtime
    return ExecutionRuntime(
        python_executable=dependencies.python_executable,
    )


def _process_command(
    dependencies: AgentDependencies,
    command: list[str],
) -> list[str]:
    runtime = _runtime_for_dependencies(dependencies)
    if runtime.runtime_type == "local":
        return command

    source = str(dependencies.workspace_root)
    return [
        str(runtime.docker_executable),
        "run",
        "--rm",
        "--network",
        "none",
        "--mount",
        f"type=bind,source={source},target=/workspace",
        "--workdir",
        "/workspace",
        "--env",
        "PYTHONPATH=/workspace/src:/workspace",
        str(runtime.container_image),
        *command,
    ]


def execute_process(
    dependencies: AgentDependencies,
    command: list[str],
    timeout_seconds: float | int,
    tool_name: str,
    *,
    action_id: str | None = None,
) -> ActionResult:
    """Execute structured argv in the workspace and normalize its result."""
    action_id = action_id or str(uuid4())
    started_at = datetime.now(UTC)

    if not command:
        return _result(
            action_id,
            tool_name,
            started_at,
            False,
            None,
            None,
            "command must not be empty",
        )
    if timeout_seconds <= 0:
        return _result(
            action_id,
            tool_name,
            started_at,
            False,
            None,
            None,
            "timeout_seconds must be greater than zero",
        )

    try:
        completed = subprocess.run(
            _process_command(dependencies, command),
            cwd=dependencies.workspace_root,
            env=workspace_process_environment(dependencies.workspace_root),
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            shell=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as timeout_error:
        stdout = _as_text(timeout_error.stdout) or None
        stderr = _as_text(timeout_error.stderr)
        error = f"command timed out after {timeout_seconds} seconds"
        if stderr:
            error = f"{error}: {stderr}"
        return _result(action_id, tool_name, started_at, False, None, stdout, error)
    except FileNotFoundError:
        return _result(
            action_id,
            tool_name,
            started_at,
            False,
            None,
            None,
            "executable was not found",
        )
    except PermissionError:
        return _result(
            action_id,
            tool_name,
            started_at,
            False,
            None,
            None,
            "process execution permission was denied",
        )
    except (OSError, ValueError) as error:
        return _result(
            action_id,
            tool_name,
            started_at,
            False,
            None,
            None,
            f"process could not be started: {error}",
        )

    stdout = completed.stdout
    stderr = completed.stderr or None
    if completed.returncode == 0:
        return _result(
            action_id,
            tool_name,
            started_at,
            True,
            completed.returncode,
            stdout,
            stderr,
        )

    return _result(
        action_id,
        tool_name,
        started_at,
        False,
        completed.returncode,
        stdout,
        stderr or f"process exited with status {completed.returncode}",
    )


def run_command(
    dependencies: AgentDependencies,
    command: list[str],
    timeout_seconds: float | int,
    *,
    action_id: str | None = None,
) -> ActionResult:
    """Execute structured argv with ``shell=False``.

    Correct: ``["git", "status", "--short"]`` or
    ``["python", "-m", "pytest", "tests/test_locales.py::Test...", "-q"]``.
    Incorrect: ``["git status --short"]``.  ``rg`` is not guaranteed in
    benchmark containers; prefer portable tools or Python when necessary.
    """
    action_id = action_id or str(uuid4())
    started_at = datetime.now(UTC)
    try:
        if command:
            command = canonicalize_command(command)
    except CommandCanonicalizationError as error:
        result = _result(
            action_id,
            "run_command",
            started_at,
            False,
            None,
            None,
            str(error),
        )
        emit_action_event(dependencies, result)
        return result
    result = execute_process(
        dependencies,
        command,
        timeout_seconds,
        "run_command",
        action_id=action_id,
    )
    emit_action_event(dependencies, result)
    return result
