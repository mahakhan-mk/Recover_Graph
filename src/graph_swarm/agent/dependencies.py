"""Typed runtime dependencies supplied to the controlled experimental agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.expanduser().resolve()
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if self.task is not None and self.task.id != self.task_id:
            raise ValueError("task.id must match task_id")

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

    def resolve_workspace_path(self, relative_path: str | Path) -> Path:
        """Resolve a path and reject traversal or symlink escapes."""
        try:
            resolved_path = (self.workspace_root / Path(relative_path)).resolve()
        except (OSError, RuntimeError) as error:
            raise WorkspacePathError(
                f"could not resolve workspace path: {relative_path!s}"
            ) from error

        if not resolved_path.is_relative_to(self.workspace_root):
            raise WorkspacePathError(f"path resolves outside workspace: {relative_path!s}")
        return resolved_path
