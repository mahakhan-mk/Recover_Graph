"""Controlled exact-text editing tool for files inside the agent workspace."""

from datetime import UTC, datetime
from uuid import uuid4

from graph_swarm.agent.dependencies import AgentDependencies, WorkspacePathError
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.domain.action import ActionResult


def _completion_time(started_at: datetime) -> datetime:
    return max(datetime.now(UTC), started_at)


def edit_file(
    dependencies: AgentDependencies,
    path: str,
    old_text: str,
    new_text: str,
    *,
    expected_replacements: int = 1,
    action_id: str | None = None,
) -> ActionResult:
    """Replace an exact number of text occurrences in one workspace file."""
    action_id = action_id or str(uuid4())
    started_at = datetime.now(UTC)
    result: ActionResult | None = None
    error: str | None = None

    try:
        if not old_text:
            raise ValueError("old_text must not be empty")
        if isinstance(expected_replacements, bool) or expected_replacements < 1:
            raise ValueError("expected_replacements must be at least 1")
        resolved_path = dependencies.resolve_workspace_path(path)
        if not resolved_path.exists():
            raise FileNotFoundError
        if not resolved_path.is_file():
            raise IsADirectoryError
        content = resolved_path.read_text(encoding="utf-8")
        occurrence_count = content.count(old_text)
        if occurrence_count != expected_replacements:
            raise ValueError(
                "exact edit expected "
                f"{expected_replacements} occurrence{'s' if expected_replacements != 1 else ''} "
                f"but found {occurrence_count}"
            )
        resolved_path.write_text(
            content.replace(old_text, new_text),
            encoding="utf-8",
        )
    except WorkspacePathError:
        result = ActionResult(
            action_id=action_id,
            tool_name="edit_file",
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
        error = "file could not be edited because permission was denied"
    except UnicodeError:
        error = "file is not valid UTF-8 text"
    except ValueError as validation_error:
        error = str(validation_error)
    except OSError:
        error = "file could not be edited"
    else:
        relative_path = resolved_path.relative_to(dependencies.workspace_root).as_posix()
        result = ActionResult(
            action_id=action_id,
            tool_name="edit_file",
            success=True,
            exit_code=None,
            output=(
                f"Edited {expected_replacements} occurrence"
                f"{'s' if expected_replacements != 1 else ''} in {relative_path}"
            ),
            error=None,
            started_at=started_at,
            completed_at=_completion_time(started_at),
        )

    if result is None:
        result = ActionResult(
            action_id=action_id,
            tool_name="edit_file",
            success=False,
            exit_code=None,
            output=None,
            error=error or "file could not be edited",
            started_at=started_at,
            completed_at=_completion_time(started_at),
        )

    emit_action_event(dependencies, result)
    return result
