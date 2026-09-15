"""Controlled structured-argv process execution tool."""

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.domain.action import ActionResult


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
    """Execute a structured command in the configured workspace."""
    result = execute_process(
        dependencies,
        command,
        timeout_seconds,
        "run_command",
        action_id=action_id,
    )
    emit_action_event(dependencies, result)
    return result
