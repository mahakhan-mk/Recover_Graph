"""Narrow Gate A1 R8 harness primitives.

This module is deliberately provider-agnostic at its public boundaries.  The
acquisition runner wires these primitives to the frozen SWE-smith objective;
unit tests can exercise the lifecycle without credentials, Docker, or Neo4j.
"""
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from graph_swarm.agent.coding_agent import AgentWallClockTimeoutError
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.stagnation import PreMutationStagnationGuard
from graph_swarm.detection.failure_detector import detect_failure
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.events import AgentEvent
from graph_swarm.graph.neo4j_repository import Neo4jRepository
from graph_swarm.integration.event_persistence import (
    persist_agent_event,
    select_recovery_event_chain,
)
from graph_swarm.memory.recovery_evidence import RepositoryMutationEvidence

R8_LINE_ENDING_POLICY = "git_index_consistent_line_endings_v2"
R8_STOPPING_POLICY = "objective_success_or_timeout_v1"
R8_PERSISTENCE_SESSION_POLICY = "short_lived_session_v1"
R8_PERSISTENCE_RETRY_POLICY = "retryable_transient_max_2_v1"
R8_MAX_PERSISTENCE_ATTEMPTS = 2
R8_MUTATING_TOOLS = frozenset({"write_file", "edit_file", "run_command", "run_tests"})


def agent_timeout_seconds_for_configuration(configuration: Any) -> float:
    """Resolve only the coding-agent budget for a validated configuration."""
    timeout = getattr(
        configuration.config,
        "agent_timeout_seconds",
        configuration.config.limits.timeout_seconds,
    )
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("agent timeout must be positive")
    return float(timeout)


def _objective_observation_payload(objective: Any) -> dict[str, object] | None:
    observations = getattr(objective, "observations", None)
    if not observations:
        return None
    latest = observations[-1]
    if dataclasses.is_dataclass(latest):
        return cast(dict[str, object], dataclasses.asdict(cast(Any, latest)))
    model_dump = getattr(latest, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return cast(dict[str, object], dumped) if isinstance(dumped, dict) else None
    return None


def objective_final_check_policy(
    revision: str,
    runtime_error: BaseException | None,
    *,
    has_repository_mutation: bool,
) -> str:
    """Return the bounded post-agent objective policy for one revision."""
    if revision == "R12":
        return "skipped_r12_requires_mutation_bound_objective"
    if revision == "R13":
        return "skipped_r13_requires_mutation_bound_objective"
    if revision == "R13b":
        return "skipped_r13b_requires_mutation_bound_objective"
    if (
        revision in {"R10", "R11"}
        and isinstance(runtime_error, AgentWallClockTimeoutError)
        and not has_repository_mutation
    ):
        return "skipped_no_repository_mutation_after_agent_timeout"
    return "performed"


def recovery_pattern_telemetry(pattern: Any, revision: str) -> dict[str, object]:
    telemetry: dict[str, object] = {
        "pattern_id": pattern.id,
        "verification_status": pattern.verification_status.value,
    }
    if revision in {"R13", "R13b"}:
        telemetry.update(
            {
                "source_tool": pattern.source_tool,
                "source_operation": pattern.source_operation,
                "applicability_tool": pattern.applicability_tool,
                "applicability_operation": pattern.applicability_operation,
            }
        )
    return telemetry


def classify_r8_acquisition(
    *,
    task_success: bool,
    complete_trusted_lineage: bool,
    pattern_created: bool,
    pattern_persisted: bool,
    pattern_embedded: bool,
    qualifying_failure_observed: bool,
) -> tuple[bool, str]:
    """Separate objective task success from complete acquisition evidence."""
    acquisition_success = bool(
        complete_trusted_lineage
        and pattern_created
        and pattern_persisted
        and pattern_embedded
    )
    if acquisition_success:
        return True, "complete_trusted_recovery_lineage_and_pattern"
    if task_success and not complete_trusted_lineage:
        return (
            False,
            "objective_success_without_complete_recovery_lineage"
            if qualifying_failure_observed
            else "no_qualifying_failure_observed",
        )
    if not complete_trusted_lineage:
        return False, "no_complete_trusted_recovery_lineage"
    if not pattern_created:
        return False, "recovery_pattern_not_created"
    if not pattern_persisted:
        return False, "recovery_pattern_not_persisted"
    return False, "recovery_pattern_not_embedded"


def r8_acquisition_readiness(
    task_records: Sequence[Mapping[str, object]],
    completed_task_ids: tuple[str, ...],
    task_ids: tuple[str, ...],
) -> tuple[str, bool, int]:
    """Return readiness using completed markers and eligible corpus state."""
    eligible_task_ids = {
        record.get("task_id")
        for record in task_records
        if isinstance(record.get("task_id"), str)
        and record.get("task_id") in task_ids
        and bool(record.get("acquisition_success"))
    }
    eligible_tasks = sum(task_id in eligible_task_ids for task_id in task_ids)
    all_completed = completed_task_ids == task_ids
    if all_completed and eligible_tasks > 0:
        return "READY_FOR_GATE_A1_RETRIEVAL_EVALUATION", True, eligible_tasks
    if all_completed:
        return "BLOCKED_GATE_A1_ACQUISITION_R8_NON_EVALUABLE", False, eligible_tasks
    return "READY_TO_RESUME_GATE_A1_ACQUISITION_R8", False, eligible_tasks


class ObjectiveSatisfied(Exception):
    """Internal controlled completion signal, never shown to the model."""


class ObjectiveEvaluator(Protocol):
    def __call__(self, task: Any, workspace: Path) -> bool:
        ...


def repository_state_fingerprint(workspace: Path) -> str:
    """Hash repository bytes, modes, and Git metadata without changing state."""
    root = workspace.resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(f"L:{relative}:{os.readlink(path)}\n".encode())
            continue
        if not path.is_file():
            continue
        try:
            metadata = path.stat()
            content = path.read_bytes()
        except OSError:
            continue
        digest.update(f"F:{relative}:{metadata.st_mode & 0o7777}:{len(content)}\n".encode())
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def git_worktree_content_fingerprint(workspace: Path) -> str:
    """Hash substantive Git-visible worktree content, excluding runtime metadata.

    Git supplies the tracked and non-ignored untracked path set.  The index is
    never hashed, and file bytes are read from the worktree, so stat refreshes
    and other Git metadata changes do not look like repository mutations.
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not enumerate Git worktree content: {result.stderr!r}"
        )
    paths = sorted(path for path in result.stdout.split(b"\0") if path)
    digest = hashlib.sha256()
    for relative_bytes in paths:
        relative = os.fsdecode(relative_bytes)
        if relative == ".git" or relative.startswith(".git/"):
            continue
        path = workspace / Path(relative)
        digest.update(b"P\0" + relative_bytes + b"\0")
        try:
            metadata = os.lstat(path)
        except OSError:
            digest.update(b"MISSING\0")
            continue
        digest.update(f"{metadata.st_mode & 0o7777}\0".encode("ascii"))
        if stat.S_ISLNK(metadata.st_mode):
            digest.update(b"L\0" + os.fsencode(os.readlink(path)) + b"\0")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            digest.update(b"O\0")
            continue
        try:
            content = path.read_bytes()
        except OSError:
            digest.update(b"MISSING\0")
            continue
        digest.update(b"F\0" + len(content).to_bytes(8, "big") + b"\0")
        digest.update(content)
    return digest.hexdigest()


def _is_retryable_neo4j(error: BaseException) -> bool:
    return isinstance(error, (SessionExpired, ServiceUnavailable, TransientError))


def _sanitized_message(error: BaseException) -> str:
    message = re.sub(
        r"(?i)(password|token|secret|api[_-]?key)\\s*[=:]\\s*[^,;\\s]+",
        r"\1=<redacted>",
        str(error),
    )
    return message[:500]


class InLoopPersistenceError(RuntimeError):
    """A trusted action could not be persisted during objective processing."""


class ShortLivedNeo4jRepository:
    """Proxy each repository operation through a fresh bounded owner.

    The proxy intentionally delegates schema and Cypher ownership to
    ``Neo4jRepository``.  A repository instance is never retained between
    calls, and retry attempts always construct a new instance.
    """


    def __init__(self, repository_factory: Callable[[], Neo4jRepository]) -> None:
        self._repository_factory = repository_factory
        self.telemetry: list[dict[str, object]] = []

    def _call(self, stage: str, method: str, *args: object, **kwargs: object) -> object:
        last_error: BaseException | None = None
        for attempt in range(1, R8_MAX_PERSISTENCE_ATTEMPTS + 1):
            repository: Neo4jRepository | None = None
            try:
                repository = self._repository_factory()
                operation = getattr(repository, method)
                return operation(*args, **kwargs)
            except Exception as error:
                retryable = _is_retryable_neo4j(error)
                retry_performed = retryable and attempt < R8_MAX_PERSISTENCE_ATTEMPTS
                self.telemetry.append(
                    {
                        "persistence_stage": stage,
                        "exception_type": type(error).__name__,
                        "sanitized_message": _sanitized_message(error),
                        "attempt_number": attempt,
                        "retryable": retryable,
                        "retry_performed": retry_performed,
                    }
                )
                last_error = error
                if not retry_performed:
                    raise
            finally:
                if repository is not None:
                    repository.close()
        assert last_error is not None
        raise last_error

    def __getattr__(self, name: str) -> Any:
        def operation(*args: object, **kwargs: object) -> object:
            return self._call(f"persist_{name.removeprefix('save_')}", name, *args, **kwargs)

        return operation


class R8ObjectiveController:
    """Mutation-aware out-of-band objective lifecycle for one agent run."""

    mutation_fingerprint = staticmethod(repository_state_fingerprint)

    def __init__(
        self,
        *,
        task: Any,
        workspace: Path,
        objective: ObjectiveEvaluator,
        dependencies: AgentDependencies,
        repository: ShortLivedNeo4jRepository | None = None,
        run: Any | None = None,
        environment: Any | None = None,
        pre_mutation_guard: PreMutationStagnationGuard | None = None,
    ) -> None:
        self.task = task
        self.workspace = workspace
        self.objective = objective
        self.dependencies = dependencies
        self.repository = repository
        self.run = run
        self.environment = environment
        self.pre_mutation_guard = pre_mutation_guard
        self._before: dict[str, str] = {}
        self.objective_checks = 0
        self.objective_result = False
        self.objective_error: Exception | None = None
        self.objective_evaluation_duration_seconds = 0.0
        self.objective_success = False
        self.objective_success_pending_evidence = False
        self.termination_reason: str | None = None
        self.last_action_persisted = False
        self.persistence_error: Exception | None = None
        self._persisted_action_ids: set[str] = set()

    def before_action(self, action: PlannedAction) -> None:
        if action.tool in R8_MUTATING_TOOLS:
            self._before[action.id] = self.mutation_fingerprint(self.workspace)

    def after_event(self, event: AgentEvent) -> None:
        before = self._before.pop(event.action_id, None)
        after = (
            self.mutation_fingerprint(self.workspace) if before is not None else None
        )
        mutated = (
            event.result.tool_name in R8_MUTATING_TOOLS
            and before is not None
            and after is not None
            and before != after
        )
        if mutated:
            action = self.dependencies.planned_action_for(event.action_id)
            if action is not None:
                self.dependencies.record_repository_mutation_evidence(
                    RepositoryMutationEvidence(
                        action_id=action.id,
                        before_fingerprint=cast(str, before),
                        after_fingerprint=cast(str, after),
                    )
                )
                if self.pre_mutation_guard is not None:
                    self.pre_mutation_guard.observe_dependencies(self.dependencies)
            self._persist_event(event)
            self.check_objective(event)
        if self.pre_mutation_guard is not None:
            self.pre_mutation_guard.after_tool_event(self.dependencies)
        if self.objective_success_pending_evidence and self._trusted_recovery_complete():
            self._persist_event(event)
            self.termination_reason = "objective_satisfied_with_trusted_recovery_evidence"
            raise ObjectiveSatisfied("trusted recovery evidence completed")

    def _persist_event(self, event: AgentEvent) -> None:
        if event.action_id in self._persisted_action_ids:
            return
        if self.repository is None or self.run is None or self.environment is None:
            self._persisted_action_ids.add(event.action_id)
            return
        planned = self.dependencies.planned_action_for(event.action_id)
        if planned is None:
            error = RuntimeError(f"missing trusted PlannedAction for {event.action_id}")
            self.persistence_error = error
            raise InLoopPersistenceError(str(error)) from error
        try:
            persist_agent_event(
                cast(Any, self.repository),
                event,
                self.task,
                self.run,
                self.environment,
                planned_action=planned,
            )
        except Exception as error:
            self.persistence_error = error
            raise InLoopPersistenceError(str(error)) from error
        self._persisted_action_ids.add(event.action_id)
        self.last_action_persisted = True

    def _trusted_recovery_complete(self) -> bool:
        return (
            select_recovery_event_chain(
                self.dependencies.events,
                self.dependencies.planned_actions,
                self.dependencies.repository_mutation_evidence,
            )
            is not None
        )

    def check_objective(self, _event: AgentEvent | None = None) -> bool:
        self.objective_checks += 1
        started = time.perf_counter()
        try:
            passed = bool(self.objective(self.task, self.workspace))
        except Exception as error:
            self.objective_error = error
            return False
        finally:
            self.objective_evaluation_duration_seconds += time.perf_counter() - started
        self.objective_result = passed
        if passed:
            self.objective_success = True
            self.objective_success_pending_evidence = True
            self.termination_reason = "objective_satisfied_pending_recovery_evidence"
        return False

    def final_check(self) -> bool:
        if self.objective_success:
            return False
        try:
            return self.check_objective()
        except ObjectiveSatisfied:
            self.termination_reason = "objective_satisfied_at_final_check"
            return True


def attach_r8_objective_controller(
    dependencies: AgentDependencies,
    controller: R8ObjectiveController,
) -> None:
    """Attach lifecycle hooks without exposing objective data to model history."""
    dependencies.before_tool_action = controller.before_action
    dependencies.after_tool_event = controller.after_event


def run_gate_a1_acquisition_r8(
    project_root: Path,
    *,
    resume_root: Path | None = None,
    max_new_tasks: int | None = None,
) -> tuple[str, Path]:
    """Execute the R8 acquisition lifecycle when explicitly invoked."""
    import json
    import uuid
    from datetime import UTC, datetime

    from experiments.sprint3 import FrozenSWEsmithObjective, _configured_runtime
    from graph_swarm.agent.pacing import ProviderRequestPacing
    from graph_swarm.research.gate_a1_acquisition import (
        ACQUISITION_TASK_IDS,
        _preflight,
        _r2_task_marker,
        _settings_for_agent,
        _task_cases,
        _validate_preflight_configuration,
        _write_json,
    )
    from graph_swarm.research.runner import load_experiment_configuration

    project_root = project_root.expanduser().resolve()
    if max_new_tasks is not None and max_new_tasks <= 0:
        raise ValueError("max_new_tasks must be positive")
    run_prefix = "acquisition-r8"
    if resume_root is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        artifact_root = (
            project_root
            / "research/evidence/results/GS-E003/gate_a1"
            / f"{run_prefix}-{stamp}"
        )
        artifact_root.mkdir(parents=True, exist_ok=False)
    else:
        artifact_root = resume_root.expanduser().resolve()
        if not artifact_root.name.startswith(f"{run_prefix}-"):
            raise ValueError("R8 resume path must use an acquisition-r8-* namespace")
        early_manifest = artifact_root / "manifest.json"
        if not early_manifest.is_file():
            raise ValueError("R8 resume root is missing manifest.json")
        early_record = json.loads(early_manifest.read_text(encoding="utf-8"))
        if not isinstance(early_record, dict) or early_record.get("run_revision") != "R8":
            raise ValueError("R8 rejects resume roots from R2-R7 or other revisions")

    configuration = _configured_runtime(
        load_experiment_configuration(
            project_root / "configs/experiments/gate_a1_acquisition_r8.yaml",
            project_root=project_root,
        ),
        project_root / "benchmark/workspaces",
        project_root / "research/evidence/workspaces",
    )
    if configuration.config.run_revision != "R8":
        raise ValueError("R8 configuration must declare run_revision=R8")
    if (
        configuration.config.limits.max_actions is not None
        or configuration.config.limits.max_requests is not None
    ):
        raise ValueError("R8 action and request limits must be explicitly disabled")
    if configuration.config.limits.timeout_seconds != 600:
        raise ValueError("R8 wall-clock timeout must remain 600 seconds")
    cases = _task_cases(configuration, ACQUISITION_TASK_IDS)
    settings = _settings_for_agent()
    _validate_preflight_configuration(
        configuration,
        settings,
        expected_coding_model="nex-agi/nex-n2.5-pro:free",
        expected_abstraction_model="cohere/north-mini-code:free",
    )
    environments, frozen_cases = _preflight(
        project_root=project_root,
        baseline_root=project_root / "benchmark/workspaces",
        execution_root=project_root / "research/evidence/workspaces",
        configuration=configuration,
        cases=cases,
    )
    objective = FrozenSWEsmithObjective(
        frozen_cases,
        environments,
        objective_coverage_policy=configuration.config.objective_coverage_policy,
        coverage_policy_selection_version=configuration.config.objective_coverage_policy_selection_version,
    )
    from graph_swarm.research.gate_a1_acquisition import _r8_configuration_hash

    configuration_hash = _r8_configuration_hash(configuration, settings, environments)
    manifest: dict[str, object] = {
        "gate": "GS-E003 / Gate A1",
        "phase": "acquisition",
        "run_revision": "R8",
        "namespace": "GS-E003/Gate-A1/acquisition-r8",
        "task_ids": list(ACQUISITION_TASK_IDS),
        "status": "BLOCKED_GATE_A1_ACQUISITION_R8",
        "configuration_hash": configuration_hash,
        "coding_model": settings.openrouter_coding_model,
        "abstraction_model": settings.openrouter_abstraction_model,
        "prompt_version": configuration.model.prompt_version,
        "limits": configuration.config.limits.model_dump(mode="json"),
        "stopping_policy": R8_STOPPING_POLICY,
        "objective_mutation_check_policy": "repository_state_fingerprint_v1",
        "workspace_line_ending_policy": R8_LINE_ENDING_POLICY,
        "persistence_session_policy": R8_PERSISTENCE_SESSION_POLICY,
        "persistence_retry_policy": R8_PERSISTENCE_RETRY_POLICY,
        "revision_reason": configuration.config.revision_reason,
        "model_visible_tool_output_chars": configuration.config.model_visible_tool_output_chars,
        "objective_coverage_policy": configuration.config.objective_coverage_policy,
        "objective_coverage_policy_selection_version": (
            configuration.config.objective_coverage_policy_selection_version
        ),
        "tasks": [],
        "errors": [],
    }
    manifest_path = artifact_root / "manifest.json"
    if resume_root is not None:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("run_revision") != "R8":
            raise ValueError("R8 rejects resume roots from R2-R7 or other revisions")
        if existing.get("configuration_hash") != configuration_hash:
            raise ValueError("R8 resume configuration hash differs")
        manifest = cast(dict[str, object], existing)
    else:
        _write_json(manifest_path, manifest)

    def repository_factory() -> Neo4jRepository:
        return Neo4jRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )

    memory = ShortLivedNeo4jRepository(repository_factory)
    memory.verify_connectivity()
    memory.ensure_recovery_pattern_vector_index()
    task_records: list[dict[str, object]] = []
    embedder: Any = None
    started_new = 0
    for case in cases:
        task_id = case.task.id
        completed_marker = _r2_task_marker(artifact_root, task_id, "completed.json")
        started_marker = _r2_task_marker(artifact_root, task_id, "started.json")
        if completed_marker.is_file():
            task_records.append(json.loads(completed_marker.read_text(encoding="utf-8")))
            continue
        if started_marker.is_file():
            task_records.append({"task_id": task_id, "resume_action": "skipped_started_no_rerun"})
            continue
        if max_new_tasks is not None and started_new >= max_new_tasks:
            break
        run_id = f"GS-E003-A1-R8-{task_id}-{uuid.uuid4().hex}"
        started_marker.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            started_marker,
            {
                "task_id": task_id,
                "run_id": run_id,
                "configuration_hash": configuration_hash,
            },
        )
        task_artifact, embedder = _run_r8_task(
            project_root=project_root,
            artifact_root=artifact_root,
            configuration=configuration,
            case=case,
            environment=environments[task_id],
            objective=objective,
            settings=settings,
            pacing=ProviderRequestPacing(),
            embedder=embedder,
            run_id=run_id,
            memory=memory,
        )
        task_records.append(task_artifact)
        _write_json(completed_marker, task_artifact)
        started_new += 1
        manifest["tasks"] = task_records
        _write_json(manifest_path, manifest)
    manifest["tasks"] = task_records
    completed_task_ids = tuple(
        task_id
        for task_id in ACQUISITION_TASK_IDS
        if _r2_task_marker(artifact_root, task_id, "completed.json").is_file()
    )
    status, corpus_ready, eligible_acquisition_tasks = r8_acquisition_readiness(
        task_records,
        completed_task_ids,
        ACQUISITION_TASK_IDS,
    )
    manifest["completed_tasks"] = len(completed_task_ids)
    manifest["completed_task_ids"] = list(completed_task_ids)
    manifest["eligible_acquisition_tasks"] = eligible_acquisition_tasks
    manifest["acquisition_corpus_ready"] = corpus_ready
    manifest["status"] = status
    if status == "BLOCKED_GATE_A1_ACQUISITION_R8_NON_EVALUABLE":
        manifest["acquisition_reason"] = "zero_eligible_recovery_patterns"
    _write_json(manifest_path, manifest)
    return str(manifest["status"]), artifact_root


def _run_r8_task(
    *,
    project_root: Path,
    artifact_root: Path,
    configuration: Any,
    case: Any,
    environment: Any,
    objective: Any,
    settings: Any,
    pacing: Any,
    embedder: Any,
    run_id: str,
    memory: ShortLivedNeo4jRepository,
    revision: str = "R8",
    condition: str = "acquisition-r8",
    controller_type: type[R8ObjectiveController] = R8ObjectiveController,
    objective_anchored_acquisition: bool = False,
    system_prompt: str | None = None,
    agent_timeout_seconds: float | None = None,
    objective_timeout_seconds: float | None = None,
    runtime_timeout_overrides: dict[str, dict[str, object]] | None = None,
    attempt_number: int | None = None,
    attempt_kind: str | None = None,
    pre_mutation_guard: PreMutationStagnationGuard | None = None,
) -> tuple[dict[str, object], Any]:
    import time
    from datetime import UTC, datetime

    from pydantic_ai import ModelSettings
    from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

    from experiments.sprint3 import _materialize_workspace
    from graph_swarm.agent.coding_agent import (
        create_coding_agent,
        run_coding_agent,
        timeout_provenance,
    )
    from graph_swarm.agent.dependencies import AgentDependencies
    from graph_swarm.domain.environment import EnvironmentContext
    from graph_swarm.domain.runs import Run
    from graph_swarm.integration.event_persistence import (
        persist_agent_event_stream,
        persist_objective_anchored_recovery,
    )
    from graph_swarm.memory.recovery_abstraction import abstract_and_persist_recovery_pattern
    from graph_swarm.memory.recovery_embeddings import (
        RecoveryPatternEmbedder,
        embed_and_persist_recovery_pattern,
    )
    from graph_swarm.research.gate_a1_acquisition import _write_json

    task = case.task
    workspace = _materialize_workspace(
        source_root=project_root / "benchmark/workspaces",
        execution_root=project_root / "research/evidence/workspaces",
        frozen_cases=objective.cases,
        condition=condition,
        task=task,
        workspace_line_ending_policy=R8_LINE_ENDING_POLICY,
    )
    run = Run(id=run_id, task_id=task.id, started_at=datetime.now(UTC))
    environment_context = EnvironmentContext(
        id=f"{run_id}-environment",
        repository=task.repository,
        runtime="docker",
        versions={"python": environment.python_version or "unknown"},
        markers={"memory_write_only": "true", "retrieval_performed": "false"},
    )
    memory.save_task(task)
    memory.save_run(run)
    memory.save_environment(environment_context)
    run_dir = artifact_root / f"GS-E003/gate_a1/{condition}" / task.id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    dependencies = AgentDependencies(
        workspace_root=workspace,
        run_id=run_id,
        task_id=task.id,
        execution_runtime=environment.agent_execution_runtime(),
    )
    controller = controller_type(
        task=task,
        workspace=workspace,
        objective=objective,
        dependencies=dependencies,
        repository=memory,
        run=run,
        environment=environment_context,
        pre_mutation_guard=pre_mutation_guard,
    )
    attach_r8_objective_controller(dependencies, controller)
    result: Any = None
    runtime_error: Exception | None = None
    persistence_error: Exception | None = None
    started = time.perf_counter()
    agent_started: float | None = None
    agent_duration_seconds = 0.0
    effective_agent_timeout_seconds = (
        agent_timeout_seconds
        if agent_timeout_seconds is not None
        else agent_timeout_seconds_for_configuration(configuration)
    )
    try:
        pre_agent_passed = False
        if objective_anchored_acquisition:
            pre_agent_passed = bool(controller.check_pre_agent_objective())
        if not objective_anchored_acquisition or (
            not pre_agent_passed
            and getattr(controller, "pre_agent_objective_error", None) is None
        ):
            step_persistence = StepPersistence(
                store=SqliteStepStore(database=run_dir / "steps.sqlite"),
                agent_name="graph_swarm_coding_agent",
                run_id=run_id,
                metadata={
                    "gate": "GS-E003 / Gate A1",
                    "revision": revision,
                    "task_id": task.id,
                },
            )
            agent = create_coding_agent(
                settings,
                capabilities=[step_persistence],
                system_prompt=system_prompt,
            )
            agent_started = time.perf_counter()
            if pre_mutation_guard is not None:
                pre_mutation_guard.start()
            result = run_coding_agent(
                agent,
                settings,
                dependencies,
                __import__(
                    "graph_swarm.research.runner", fromlist=["build_task_prompt"]
                ).build_task_prompt(task),
                max_actions=None,
                max_requests=None,
                timeout_seconds=effective_agent_timeout_seconds,
                model_settings=cast(ModelSettings, configuration.model.settings),
                request_pacing=pacing,
                disable_request_limit=True,
                pre_mutation_guard=pre_mutation_guard,
            )
    except ObjectiveSatisfied:
        pass
    except InLoopPersistenceError as error:
        persistence_error = error
    except Exception as error:
        runtime_error = error
    finally:
        if agent_started is not None:
            agent_duration_seconds = time.perf_counter() - agent_started
    objective_final_check = "not_needed_objective_already_satisfied"
    if objective_anchored_acquisition and getattr(
        controller, "pre_agent_objective_passed", None
    ):
        objective_final_check = "not_needed_pre_agent_objective_already_satisfied"
    elif objective_anchored_acquisition and getattr(
        controller, "pre_agent_objective_error", None
    ) is not None:
        objective_final_check = "not_needed_pre_agent_objective_error"
    elif objective_anchored_acquisition:
        objective_final_check = objective_final_check_policy(
            revision,
            runtime_error,
            has_repository_mutation=bool(dependencies.repository_mutation_evidence),
        )
    elif not controller.objective_success:
        objective_final_check = objective_final_check_policy(
            revision,
            runtime_error,
            has_repository_mutation=bool(dependencies.repository_mutation_evidence),
        )
        if objective_final_check == "performed":
            controller.final_check()
    persistence_error: Exception | None = persistence_error or controller.persistence_error
    recovery: dict[str, object] | None = None
    recovery_pattern_created = False
    recovery_pattern_persisted = False
    recovery_pattern_embedded = False
    patterns: list[dict[str, object]] = []
    try:
        if objective_anchored_acquisition:
            persisted = None
            if (
                getattr(controller, "pre_agent_objective_passed", None) is False
                and getattr(controller, "pre_agent_objective_error", None) is None
                and getattr(controller, "objective_success_trigger_action_id", None)
                is not None
                and getattr(controller, "objective_evidence_error", None) is None
            ):
                persisted = persist_objective_anchored_recovery(
                    cast(Any, memory),
                    dependencies.events,
                    task,
                    run,
                    environment_context,
                    pre_agent_objective=controller.pre_agent_objective_anchor(),
                    post_mutation_objective=controller.post_mutation_objective_anchor(),
                    planned_actions=dependencies.planned_actions,
                    mutation_evidence=dependencies.repository_mutation_evidence,
                    objective_success_trigger_action_id=cast(
                        str, controller.objective_success_trigger_action_id
                    ),
                )
        else:
            persisted = persist_agent_event_stream(
                cast(Any, memory),
                dependencies.events,
                task,
                run,
                environment_context,
                planned_actions=dependencies.planned_actions,
                require_trusted_planned_actions=True,
                mutation_evidence=dependencies.repository_mutation_evidence,
            )
        if persisted is not None:
            failure, resolution, outcome = persisted
            recovery = {
                "failure_id": failure.id,
                "resolution_id": resolution.id,
                "outcome_id": outcome.id,
            }
            lineage = memory.get_recovery_evidence(failure.id)
            if revision in {"R13", "R13b"}:
                from graph_swarm.memory.recovery_abstraction import (
                    OBJECTIVE_ANCHORED_RECOVERY_EVIDENCE_SOURCE,
                )

                lineage = lineage.model_copy(
                    update={
                        "recovery_evidence_source": (
                            OBJECTIVE_ANCHORED_RECOVERY_EVIDENCE_SOURCE
                        ),
                        "trusted_recovery_action_id": getattr(
                            controller,
                            "objective_success_trigger_action_id",
                            None,
                        ),
                    }
                )
            pattern = abstract_and_persist_recovery_pattern(
                lineage,
                cast(Any, memory),
                settings,
            )
            recovery_pattern_created = True
            recovery_pattern_persisted = True
            if embedder is None:
                embedder = RecoveryPatternEmbedder()
            embedded = embed_and_persist_recovery_pattern(
                pattern,
                cast(Any, memory),
                embedder,
            )
            recovery_pattern_embedded = embedded.embedding is not None
            patterns.append(recovery_pattern_telemetry(embedded, revision))
    except Exception as error:
        persistence_error = persistence_error or error
    task_success = controller.objective_success
    acquisition_success = bool(
        recovery is not None
        and recovery_pattern_created
        and recovery_pattern_persisted
        and recovery_pattern_embedded
    )
    acquisition_success, acquisition_reason = classify_r8_acquisition(
        task_success=task_success,
        complete_trusted_lineage=recovery is not None,
        pattern_created=recovery_pattern_created,
        pattern_persisted=recovery_pattern_persisted,
        pattern_embedded=recovery_pattern_embedded,
        qualifying_failure_observed=any(
            detect_failure(
                event,
                dependencies.planned_action_for(event.action_id),
            )
            is not None
            for event in dependencies.events
        ),
    )
    total_task_duration_seconds = time.perf_counter() - started
    artifact: dict[str, object] = {
        "task_id": task.id,
        "run_id": run_id,
        "status": (
            "acquired"
            if acquisition_success
            else "task_succeeded_acquisition_not_evaluable"
            if task_success
            else "blocked_runtime_or_persistence_error"
            if persistence_error or runtime_error
            else "objective_not_satisfied"
        ),
        "duration_seconds": total_task_duration_seconds,
        "task_success": task_success,
        "acquisition_success": acquisition_success,
        "acquisition_reason": acquisition_reason,
        "termination_reason": controller.termination_reason
        or (
            "pre_mutation_stagnation"
            if runtime_error is not None
            and getattr(runtime_error, "termination_reason", None)
            == "pre_mutation_stagnation"
            else
            "wall_clock_timeout"
            if runtime_error is not None
            and getattr(runtime_error, "timeout_layer", None) == "agent_wall_clock"
            else "agent_returned" if runtime_error is None else "runtime_error"
        ),
        "timeout_seconds": configuration.config.limits.timeout_seconds,
        "agent_error": (
            None
            if task_success
            else None if runtime_error is None else type(runtime_error).__name__
        ),
        "agent_runtime_error": None if runtime_error is None else type(runtime_error).__name__,
        "objective_checks": controller.objective_checks,
        "objective_error": (
            None
            if controller.objective_error is None
            else type(controller.objective_error).__name__
        ),
        "persistence_error": (
            None if persistence_error is None else type(persistence_error).__name__
        ),
        "persistence_telemetry": memory.telemetry,
        "counts": {
            "events": len(dependencies.events),
            "recoveries": 0 if recovery is None else 1,
            "patterns": len(patterns),
        },
        "recovery_evidence": {
            "complete_trusted_lineage": recovery is not None,
            "pattern_created": recovery_pattern_created,
            "pattern_persisted": recovery_pattern_persisted,
            "pattern_embedded": recovery_pattern_embedded,
        },
        "recovery_lineage": recovery,
        "patterns": patterns,
        "events": [event.model_dump(mode="json") for event in dependencies.events],
        "agent_output": None if result is None else result.output,
    }
    if revision == "R13b":
        artifact.update(
            {
                "run_revision": revision,
                "config_version": configuration.config.config_version,
                "revision_reason": configuration.config.revision_reason,
                "prompt_version": configuration.model.prompt_version,
                "coding_model": settings.openrouter_coding_model,
            }
        )
    if objective_anchored_acquisition:
        pre_error = getattr(controller, "pre_agent_objective_error", None)
        pre_passed = getattr(controller, "pre_agent_objective_passed", None)
        if pre_passed is True:
            artifact["status"] = "invalid_acquisition_baseline"
            artifact["acquisition_success"] = False
            artifact["acquisition_reason"] = "pre_agent_objective_already_satisfied"
        elif pre_error is not None:
            artifact["status"] = "blocked_pre_agent_objective_error"
            artifact["acquisition_success"] = False
            artifact["acquisition_reason"] = "pre_agent_objective_error"
        elif getattr(controller, "objective_evidence_error", None) is not None:
            artifact["status"] = "blocked_objective_evidence_error"
            artifact["acquisition_success"] = False
            artifact["acquisition_reason"] = "objective_evidence_error"
        artifact.update(
            {
                "complete_trusted_lineage": recovery is not None,
                "pattern_created": recovery_pattern_created,
                "pattern_persisted": recovery_pattern_persisted,
                "pattern_embedded": recovery_pattern_embedded,
                "pre_agent_objective_checked": bool(
                    getattr(controller, "pre_agent_objective_checked", False)
                ),
                "pre_agent_objective_passed": pre_passed,
                "pre_agent_objective_error": (
                    None if pre_error is None else type(pre_error).__name__
                ),
                "objective_evidence_error": (
                    None
                    if getattr(controller, "objective_evidence_error", None) is None
                    else type(controller.objective_evidence_error).__name__
                ),
                "pre_agent_objective_observation": getattr(
                    controller, "pre_agent_objective_observation", None
                ),
                "objective_success_trigger_action_id": getattr(
                    controller, "objective_success_trigger_action_id", None
                ),
                "post_mutation_objective_observation": getattr(
                    controller, "post_mutation_objective_observation", None
                ),
                "recovery_evidence_source": (
                    "objective_anchored_v1" if recovery is not None else None
                ),
            }
        )
    if revision in {"R9", "R10", "R11", "R12", "R13", "R13b"}:
        artifact.update(
            {
                "agent_timeout_seconds": effective_agent_timeout_seconds,
                "objective_timeout_seconds": (
                    objective_timeout_seconds
                    if objective_timeout_seconds is not None
                    else getattr(configuration.config, "objective_timeout_seconds", None)
                ),
                "agent_duration_seconds": agent_duration_seconds,
                "objective_evaluation_duration_seconds": (
                    controller.objective_evaluation_duration_seconds
                ),
                "total_task_duration_seconds": total_task_duration_seconds,
                "command_argv_policy": getattr(
                    configuration.config, "command_argv_policy", None
                ),
                "objective_final_check": objective_final_check,
                "agent_runtime_error_message": (
                    None
                    if runtime_error is None
                    else _sanitized_message(runtime_error)
                ),
                "objective_error_message": (
                    None
                    if controller.objective_error is None
                    else _sanitized_message(controller.objective_error)
                ),
                "objective_observation": _objective_observation_payload(objective),
                "timeout_provenance": timeout_provenance(runtime_error),
            }
        )
        if runtime_timeout_overrides is not None:
            artifact["runtime_timeout_overrides"] = runtime_timeout_overrides
        if pre_mutation_guard is not None:
            artifact.update(pre_mutation_guard.provenance())
        if attempt_number is not None:
            artifact["attempt"] = attempt_number
        if attempt_kind is not None:
            artifact["kind"] = attempt_kind
    _write_json(run_dir / "acquisition.json", artifact)
    return artifact, embedder


__all__ = [
    "ObjectiveSatisfied",
    "InLoopPersistenceError",
    "classify_r8_acquisition",
    "R8ObjectiveController",
    "R8_LINE_ENDING_POLICY",
    "R8_MAX_PERSISTENCE_ATTEMPTS",
    "R8_MUTATING_TOOLS",
    "R8_PERSISTENCE_RETRY_POLICY",
    "R8_PERSISTENCE_SESSION_POLICY",
    "R8_STOPPING_POLICY",
    "ShortLivedNeo4jRepository",
    "attach_r8_objective_controller",
    "git_worktree_content_fingerprint",
    "objective_final_check_policy",
    "repository_state_fingerprint",
    "r8_acquisition_readiness",
]
