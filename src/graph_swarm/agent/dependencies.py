"""Typed runtime dependencies supplied to the controlled experimental agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Literal

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.behavior import BehaviorChangeEvidence
from graph_swarm.domain.events import AdviceEvent, AgentEvent

if TYPE_CHECKING:
    from graph_swarm.advisory.service import AdvisoryService
    from graph_swarm.domain.environment import EnvironmentContext
    from graph_swarm.domain.tasks import Task
    from graph_swarm.research.artifacts import JsonlResearchArtifactWriter


class WorkspacePathError(ValueError):
    """Raised when a path resolves outside the configured workspace."""


@dataclass(frozen=True)
class ExecutionRuntime:
    """Process runtime used by agent tools.

    A local runtime executes its command directly. A Docker runtime keeps the
    host Docker CLI separate from the Python executable that exists in the
    prepared container.
    """

    runtime_type: Literal["local", "docker"] = "local"
    python_executable: Path | None = None
    docker_executable: Path | None = None
    container_image: str | None = None
    container_python_executable: str | None = None

    def __post_init__(self) -> None:
        if self.runtime_type == "local":
            if self.docker_executable is not None:
                raise ValueError("local runtimes must not define docker_executable")
            return
        if self.python_executable is not None:
            raise ValueError("docker runtimes must not define python_executable")
        if self.docker_executable is None:
            raise ValueError("docker runtimes require docker_executable")
        if not self.container_image:
            raise ValueError("docker runtimes require container_image")
        if not self.container_python_executable:
            raise ValueError("docker runtimes require container_python_executable")


@dataclass
class AgentDependencies:
    """Minimal in-memory runtime context shared by future agent tools."""

    workspace_root: Path
    run_id: str
    task_id: str
    events: list[AgentEvent] = field(default_factory=lambda: list[AgentEvent]())
    planned_actions: dict[str, PlannedAction] = field(
        default_factory=lambda: dict[str, PlannedAction]()
    )
    task: Task | None = None
    environment: EnvironmentContext | None = None
    advisory_service: AdvisoryService | None = None
    artifact_writer: JsonlResearchArtifactWriter | None = None
    python_executable: Path | None = None
    execution_runtime: ExecutionRuntime | None = None
    advice_events: list[AdviceEvent] = field(default_factory=lambda: list[AdviceEvent]())
    behavior_evidence: list[BehaviorChangeEvidence] = field(
        default_factory=lambda: list[BehaviorChangeEvidence]()
    )
    advisory_errors: list[str] = field(default_factory=lambda: list[str]())
    artifact_errors: list[str] = field(default_factory=lambda: list[str]())
    _advised_action_keys: set[str] = field(default_factory=lambda: set[str](), repr=False)
    _pending_advice_events: list[AdviceEvent] = field(
        default_factory=lambda: list[AdviceEvent](), repr=False
    )
    oracle_advice_issued: bool = field(default=False, repr=False)
    task_start_guidance_evaluated: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.expanduser().resolve()
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if self.task is not None and self.task.id != self.task_id:
            raise ValueError("task.id must match task_id")

    def record_planned_action(self, action: PlannedAction) -> None:
        """Retain the exact action created immediately before tool execution.

        Planned actions are scoped to this dependency instance, which in turn
        represents one task/run. Re-recording the same action is idempotent;
        attempting to reuse its ID for different action content is rejected.
        """
        if action.run_id != self.run_id:
            raise ValueError("planned action run_id must match dependencies.run_id")
        if action.task_id != self.task_id:
            raise ValueError("planned action task_id must match dependencies.task_id")
        existing = self.planned_actions.get(action.id)
        if existing is not None and existing != action:
            raise ValueError(f"planned action id already has different content: {action.id}")
        self.planned_actions[action.id] = action

    def planned_action_for(self, action_id: str) -> PlannedAction | None:
        """Return the trusted runtime action for one completed action ID."""
        return self.planned_actions.get(action_id)

    def has_advised_action(self, action_key: str) -> bool:
        return action_key in self._advised_action_keys

    def remember_advised_action(self, action_key: str) -> None:
        self._advised_action_keys.add(action_key)

    def queue_advice_event(self, event: AdviceEvent) -> None:
        self._pending_advice_events.append(event)

    def record_advice_event(self, event: AdviceEvent) -> None:
        self.advice_events.append(event)
        if self.artifact_writer is not None:
            try:
                self.artifact_writer.write_advice_event(event)
            except Exception as error:  # noqa: BLE001 - retain in-memory evidence
                self.artifact_errors.append(f"advice artifact failed: {error}")

    def record_behavior_evidence(
        self,
        evidence: BehaviorChangeEvidence,
        subsequent_event: AgentEvent | None = None,
    ) -> None:
        self.behavior_evidence.append(evidence)
        if self.artifact_writer is None:
            return
        try:
            self.artifact_writer.write_behavior_evidence(evidence)
            if subsequent_event is not None:
                self.artifact_writer.write_subsequent_outcome(
                    evidence.advice_event_id,
                    subsequent_event,
                )
        except Exception as error:  # noqa: BLE001 - retain in-memory evidence
            self.artifact_errors.append(f"behavior artifact failed: {error}")

    def take_pending_advice_events(self) -> tuple[AdviceEvent, ...]:
        pending = tuple(self._pending_advice_events)
        self._pending_advice_events.clear()
        return pending

    def resolve_workspace_path(
        self,
        relative_path: str | Path,
    ) -> Path:
        """Resolve a workspace path and reject traversal or symlink escapes.

        Docker commands expose the task workspace at ``/workspace``. Translate
        that container path to the host-side workspace before applying the
        normal containment checks.
        """
        path_to_resolve: str | Path = relative_path
        is_container_workspace_path = False
        if self._uses_docker_runtime():
            container_path = PurePosixPath(str(relative_path))
            workspace_namespace = PurePosixPath("/workspace")
            try:
                relative_posix = container_path.relative_to(workspace_namespace)
                path_to_resolve = Path(*relative_posix.parts)
            except ValueError:
                pass
            else:
                is_container_workspace_path = True

        if (
            self._uses_docker_runtime()
            and not is_container_workspace_path
            and self._is_absolute_or_drive_path(path_to_resolve)
        ):
            raise WorkspacePathError(f"absolute path is not allowed: {relative_path!s}")

        try:
            resolved_path = (self.workspace_root / Path(path_to_resolve)).resolve()
        except (OSError, RuntimeError) as error:
            raise WorkspacePathError(
                f"could not resolve workspace path: {relative_path!s}"
            ) from error

        if not resolved_path.is_relative_to(self.workspace_root):
            raise WorkspacePathError(f"path resolves outside workspace: {relative_path!s}")
        return resolved_path

    def _uses_docker_runtime(self) -> bool:
        return (
            self.execution_runtime is not None
            and self.execution_runtime.runtime_type == "docker"
        )

    @staticmethod
    def _is_absolute_or_drive_path(path: str | Path) -> bool:
        path_text = str(path)
        return (
            Path(path_text).is_absolute()
            or PurePosixPath(path_text).is_absolute()
            or bool(PureWindowsPath(path_text).anchor)
        )
