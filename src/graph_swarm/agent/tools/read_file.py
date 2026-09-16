"""Controlled workspace text-file reading tool."""

from datetime import UTC, datetime
from uuid import uuid4

from graph_swarm.agent.dependencies import AgentDependencies, WorkspacePathError
from graph_swarm.agent.hooks import emit_action_event
from graph_swarm.domain.action import ActionResult

DEFAULT_READ_FILE_OFFSET = 0
DEFAULT_READ_FILE_LENGTH = 200
MAX_READ_FILE_LENGTH = 400
MAX_READ_FILE_CHARS = 100_000


def _completion_time(started_at: datetime) -> datetime:
    return max(datetime.now(UTC), started_at)


def read_file(
    dependencies: AgentDependencies,
    path: str,
    *,
    offset: int = DEFAULT_READ_FILE_OFFSET,
    length: int = DEFAULT_READ_FILE_LENGTH,
    action_id: str | None = None,
) -> ActionResult:
    """Read a bounded, line-based page from a UTF-8 text file in the workspace."""
    action_id = action_id or str(uuid4())
    started_at = datetime.now(UTC)
    result: ActionResult | None = None
    error: str | None = None

    try:
        if isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be greater than or equal to zero")
        if isinstance(length, bool) or length <= 0:
            raise ValueError("length must be greater than zero")
        if length > MAX_READ_FILE_LENGTH:
            raise ValueError(f"length must not exceed {MAX_READ_FILE_LENGTH} lines")
        resolved_path = dependencies.resolve_workspace_path(path)
        if not resolved_path.exists():
            raise FileNotFoundError
        if not resolved_path.is_file():
            raise IsADirectoryError
        lines = resolved_path.read_text(encoding="utf-8").splitlines(keepends=True)
        selected_lines = lines[offset : offset + length]
        content = "".join(selected_lines)
        output = _page_output(
            content,
            offset=offset,
            length=length,
            total_lines=len(lines),
        )
        if len(output) > MAX_READ_FILE_CHARS:
            raise ValueError(
                f"requested content exceeds the {MAX_READ_FILE_CHARS}-character output bound"
            )
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
    except ValueError as validation_error:
        error = str(validation_error)
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


def _page_output(content: str, *, offset: int, length: int, total_lines: int) -> str:
    """Add continuation metadata when a request is paginated or truncated."""
    end_offset = min(offset + length, total_lines)
    has_more = end_offset < total_lines
    request_was_paginated = (
        offset != DEFAULT_READ_FILE_OFFSET or length != DEFAULT_READ_FILE_LENGTH
    )
    if not has_more and not request_was_paginated:
        return content

    if offset >= total_lines or end_offset <= offset:
        returned_range = "none"
    else:
        returned_range = f"{offset + 1}-{end_offset}"
    next_offset = str(end_offset) if has_more else "none"
    metadata = (
        f"[read_file metadata: returned_lines={returned_range}; "
        f"total_lines={total_lines}; next_offset={next_offset}]"
    )
    return f"{metadata}\n{content}"
