"""Typed runtime dependencies supplied to the controlled experimental agent."""

from dataclasses import dataclass, field
from pathlib import Path

from graph_swarm.domain.events import AgentEvent


class WorkspacePathError(ValueError):
    """Raised when a path resolves outside the configured workspace."""


@dataclass
class AgentDependencies:
    """Minimal in-memory runtime context shared by future agent tools."""

    workspace_root: Path
    run_id: str
    task_id: str
    events: list[AgentEvent] = field(default_factory=lambda: list[AgentEvent]())

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.expanduser().resolve()
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")

    def resolve_workspace_path(self, relative_path: str | Path) -> Path:
        """Resolve a path and reject traversal or symlink escapes."""
        try:
            resolved_path = (self.workspace_root / Path(relative_path)).resolve()
        except (OSError, RuntimeError) as error:
            raise WorkspacePathError(
                f"could not resolve workspace path: {relative_path!s}"
            ) from error

        if not resolved_path.is_relative_to(self.workspace_root):
            raise WorkspacePathError(
                f"path resolves outside workspace: {relative_path!s}"
            )
        return resolved_path
