"""Controlled workspace text-file writing tool."""

from datetime import UTC, datetime
from uuid import uuid4

from graph_swarm.agent.dependencies import AgentDependencies, WorkspacePathError
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.domain.action import ActionResult


def _completion_time(started_at: datetime) -> datetime:
    return max(datetime.now(UTC), started_at)


def write_file(
    dependencies: AgentDependencies,
    path: str,
    content: str,
) -> ActionResult:
    """Write UTF-8 text to a file inside the configured workspace."""
    action_id = str(uuid4())
    started_at = datetime.now(UTC)
    result: ActionResult | None = None
    error: str | None = None

    try:
        resolved_path = dependencies.resolve_workspace_path(path)
        resolved_path.write_text(content, encoding="utf-8")
    except WorkspacePathError:
        result = ActionResult(
            action_id=action_id,
            tool_name="write_file",
            success=False,
            exit_code=None,
            output=None,
            error="path is outside the workspace",
            started_at=started_at,
            completed_at=_completion_time(started_at),
        )
    except FileNotFoundError:
        error = "parent directory does not exist"
    except IsADirectoryError:
        error = "path is not a regular file"
    except PermissionError:
        error = "file could not be written because permission was denied"
    except OSError:
        error = "file could not be written"
    else:
        relative_path = resolved_path.relative_to(dependencies.workspace_root).as_posix()
        result = ActionResult(
            action_id=action_id,
            tool_name="write_file",
            success=True,
            exit_code=None,
            output=f"Wrote {len(content)} characters to {relative_path}",
            error=None,
            started_at=started_at,
            completed_at=_completion_time(started_at),
        )

    if result is None:
        result = ActionResult(
            action_id=action_id,
            tool_name="write_file",
            success=False,
            exit_code=None,
            output=None,
            error=error or "file could not be written",
            started_at=started_at,
            completed_at=_completion_time(started_at),
        )

    emit_action_event(dependencies, result)
    return result
