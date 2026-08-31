"""Controlled workspace text-file reading tool."""

from datetime import UTC, datetime
from uuid import uuid4

from graph_swarm.agent.dependencies import AgentDependencies, WorkspacePathError
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.domain.action import ActionResult


def _completion_time(started_at: datetime) -> datetime:
    return max(datetime.now(UTC), started_at)


def read_file(
    dependencies: AgentDependencies,
    path: str,
    *,
    action_id: str | None = None,
) -> ActionResult:
    """Read a UTF-8 text file inside the configured workspace."""
    action_id = action_id or str(uuid4())
    started_at = datetime.now(UTC)
    result: ActionResult | None = None
    error: str | None = None

    try:
        resolved_path = dependencies.resolve_workspace_path(path)
        if not resolved_path.exists():
            raise FileNotFoundError
        if not resolved_path.is_file():
            raise IsADirectoryError
        output = resolved_path.read_text(encoding="utf-8")
    except WorkspacePathError:
        result = ActionResult(
            action_id=action_id,
            tool_name="read_file",
            success=False,
            exit_code=None,
            output=None,
            error="path is outside the workspace",
            started_at=started_at,
            completed_at=_completion_time(started_at),
        )
    except FileNotFoundError:
        error = "file does not exist"
    except IsADirectoryError:
        error = "path is not a regular file"
    except PermissionError:
        error = "file is not readable"
    except UnicodeError:
        error = "file is not valid UTF-8 text"
    except OSError:
        error = "file could not be read"
    else:
        result = ActionResult(
            action_id=action_id,
            tool_name="read_file",
            success=True,
            exit_code=None,
            output=output,
            error=None,
            started_at=started_at,
            completed_at=_completion_time(started_at),
        )

    if result is None:
        result = ActionResult(
            action_id=action_id,
            tool_name="read_file",
            success=False,
            exit_code=None,
            output=None,
            error=error or "file could not be read",
            started_at=started_at,
            completed_at=_completion_time(started_at),
        )

    emit_action_event(dependencies, result)
    return result
