"""Gate A1 R12 objective-anchored acquisition."""
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from graph_swarm.integration.event_persistence import ObjectiveAnchor
from graph_swarm.research.gate_a1_acquisition import (
    ACQUISITION_TASK_IDS,
    R9_AGENT_TIMEOUT_SECONDS,
    R9_OBJECTIVE_TIMEOUT_SECONDS,
    R9_REPOSITORY_MUTATION_EVIDENCE_POLICY,
    R10_COMMAND_ARGV_POLICY,
    R12_CONFIG,
    R12_EXPECTED_ABSTRACTION_MODEL,
    R12_EXPECTED_CODING_MODEL,
    R12_NAMESPACE,
    R12_OBJECTIVE_MUTATION_CHECK_POLICY,
    R12_RECOVERY_EVENT_SEMANTICS,
    R12_REVISION_REASON,
    R12_STOPPING_POLICY,
    _preflight,
    _r2_task_marker,
    _r8_configuration_hash,
    _settings_for_agent,
    _task_cases,
    _validate_preflight_configuration,
    _validate_r12_harness_configuration,
    _write_json,
)
from graph_swarm.research.gate_a1_r8 import (
    R8_LINE_ENDING_POLICY,
    R8_PERSISTENCE_RETRY_POLICY,
    R8_PERSISTENCE_SESSION_POLICY,
    ShortLivedNeo4jRepository,
    _run_r8_task,
    r8_acquisition_readiness,
)
from graph_swarm.research.gate_a1_r11 import R11ObjectiveController


class R12ObjectiveController(R11ObjectiveController):
    """Require a failed baseline and bind success to one observed mutation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pre_agent_objective_checked = False
        self.pre_agent_objective_passed: bool | None = None
        self.pre_agent_objective_error: Exception | None = None
        self.pre_agent_objective_observation: dict[str, object] | None = None
        self.pre_agent_objective_started_at: datetime | None = None
        self.pre_agent_objective_completed_at: datetime | None = None
        self.objective_evidence_error: R12ObjectiveEvidenceError | None = None
        self.objective_success_trigger_action_id: str | None = None
        self.post_mutation_objective_observation: dict[str, object] | None = None
        self.post_mutation_objective_started_at: datetime | None = None
        self.post_mutation_objective_completed_at: datetime | None = None

    def check_pre_agent_objective(self) -> bool:
        """Evaluate the untouched workspace before creating the coding agent."""
        self.pre_agent_objective_checked = True
        started_at = datetime.now(UTC)
        self.pre_agent_objective_started_at = started_at
        started = time.perf_counter()
        previous_count = _objective_observation_count(self.objective)
        try:
            passed = bool(self.objective(self.task, self.workspace))
            observation = _fresh_objective_observation_payload(
                self.objective,
                previous_count,
            )
            self.pre_agent_objective_observation = observation
            self.pre_agent_objective_passed = passed
            _validate_objective_evidence(
                passed,
                observation,
                phase="pre_agent",
            )
        except Exception as error:
            if isinstance(error, R12ObjectiveEvidenceError):
                self.objective_evidence_error = error
                self.pre_agent_objective_error = error
                self.termination_reason = "pre_agent_objective_not_task_failure"
            else:
                self.pre_agent_objective_error = error
                self.termination_reason = "pre_agent_objective_error"
            self.pre_agent_objective_passed = False
            return False
        finally:
            self.objective_evaluation_duration_seconds += time.perf_counter() - started
            self.pre_agent_objective_completed_at = datetime.now(UTC)
        if passed:
            self.termination_reason = "pre_agent_objective_already_satisfied"
        return passed

    def check_objective(self, event: Any | None = None) -> bool:
        """Record post-mutation success only against the triggering event."""
        self.objective_checks += 1
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        previous_count = _objective_observation_count(self.objective)
        try:
            passed = bool(self.objective(self.task, self.workspace))
            observation = _fresh_objective_observation_payload(
                self.objective,
                previous_count,
            )
            self.post_mutation_objective_observation = observation
            _validate_objective_evidence(
                passed,
                observation,
                phase="post_mutation",
            )
        except Exception as error:
            if isinstance(error, R12ObjectiveEvidenceError):
                self.objective_evidence_error = error
                self.termination_reason = "post_mutation_objective_not_task_failure"
            else:
                self.objective_error = error
            return False
        finally:
            self.objective_evaluation_duration_seconds += time.perf_counter() - started
            completed_at = datetime.now(UTC)
        self.objective_result = passed
        if not passed:
            return False
        if event is None:
            self.objective_evidence_error = R12ObjectiveEvidenceError(
                "R12 objective success is not bound to a repository mutation"
            )
            self.termination_reason = "post_mutation_objective_unbound"
            return False
        self.objective_success = True
        self.objective_success_trigger_action_id = event.action_id
        self.post_mutation_objective_started_at = started_at
        self.post_mutation_objective_completed_at = completed_at
        self.objective_success_pending_evidence = True
        self.termination_reason = "objective_satisfied_pending_recovery_evidence"
        return False

    def _trusted_recovery_complete(self) -> bool:
        trigger = self.objective_success_trigger_action_id
        return bool(
            self.pre_agent_objective_checked
            and self.pre_agent_objective_passed is False
            and self.objective_success
            and self.objective_evidence_error is None
            and trigger is not None
            and self.dependencies.repository_mutation_evidence_for(trigger) is not None
        )

    def _anchor(
        self,
        *,
        passed: bool,
        started_at: datetime | None,
        completed_at: datetime | None,
        observation: dict[str, object] | None,
    ) -> ObjectiveAnchor:
        if started_at is None or completed_at is None:
            raise ValueError("objective anchor timestamps are unavailable")
        return ObjectiveAnchor(
            run_id=self.dependencies.run_id,
            task_id=self.task.id,
            passed=passed,
            return_code=_return_code(observation, default=0 if passed else 1),
            status=_observation_status(observation),
            started_at=started_at,
            completed_at=completed_at,
        )

    def pre_agent_objective_anchor(self) -> ObjectiveAnchor:
        if not self.pre_agent_objective_checked or self.pre_agent_objective_error is not None:
            raise ValueError("R12 pre-agent objective failure anchor is unavailable")
        return self._anchor(
            passed=bool(self.pre_agent_objective_passed),
            started_at=self.pre_agent_objective_started_at,
            completed_at=self.pre_agent_objective_completed_at,
            observation=self.pre_agent_objective_observation,
        )

    def post_mutation_objective_anchor(self) -> ObjectiveAnchor:
        if self.objective_success_trigger_action_id is None:
            raise ValueError("R12 post-mutation objective anchor is unavailable")
        return self._anchor(
            passed=True,
            started_at=self.post_mutation_objective_started_at,
            completed_at=self.post_mutation_objective_completed_at,
            observation=self.post_mutation_objective_observation,
        )


def _return_code(observation: dict[str, object] | None, *, default: int) -> int:
    value = None if observation is None else observation.get("return_code")
    return value if isinstance(value, int) else default


class R12ObjectiveEvidenceError(RuntimeError):
    """Raised when one R12 objective call lacks valid fresh evidence."""


def _objective_observation_count(objective: Any) -> int:
    observations = getattr(objective, "observations", None)
    if observations is None:
        return 0
    try:
        return len(observations)
    except TypeError as error:
        raise R12ObjectiveEvidenceError(
            "R12 objective observations boundary is not sized"
        ) from error


def _fresh_objective_observation_payload(
    objective: Any,
    previous_count: int,
) -> dict[str, object]:
    observations = getattr(objective, "observations", None)
    if observations is None or len(observations) <= previous_count:
        raise R12ObjectiveEvidenceError(
            "R12 objective evaluation produced no fresh observation"
        )
    latest = observations[-1]
    if dataclasses.is_dataclass(latest):
        payload = dataclasses.asdict(cast(Any, latest))
    else:
        model_dump = getattr(latest, "model_dump", None)
        payload = model_dump(mode="json") if callable(model_dump) else None
    if not isinstance(payload, dict):
        raise R12ObjectiveEvidenceError(
            "R12 objective fresh observation is not serializable"
        )
    return cast(dict[str, object], payload)


def _validate_objective_evidence(
    passed: bool,
    observation: dict[str, object],
    *,
    phase: str,
) -> None:
    status = observation.get("status")
    return_code = observation.get("return_code")
    if passed and status == "passed" and return_code == 0:
        return
    if (
        not passed
        and status == "test_failure"
        and isinstance(return_code, int)
        and return_code != 0
    ):
        return
    raise R12ObjectiveEvidenceError(
        f"R12 {phase} objective evidence is not a task result: "
        f"status={str(status)[:80]!r}; return_code={return_code!r}"
    )


def _observation_status(observation: dict[str, object] | None) -> str:
    status = None if observation is None else observation.get("status")
    if not isinstance(status, str) or not status:
        raise R12ObjectiveEvidenceError("R12 objective observation status is missing")
    return status


def run_gate_a1_acquisition_r12(
    project_root: Path,
    *,
    resume_root: Path | None = None,
    max_new_tasks: int | None = None,
) -> tuple[str, Path]:
    """Execute R12 with a fresh namespace and objective-anchored lifecycle."""
    from experiments.sprint3 import FrozenSWEsmithObjective, _configured_runtime
    from graph_swarm.agent.pacing import ProviderRequestPacing
    from graph_swarm.graph.neo4j_repository import Neo4jRepository
    from graph_swarm.research.runner import load_experiment_configuration

    project_root = project_root.expanduser().resolve()
    if max_new_tasks is not None and max_new_tasks <= 0:
        raise ValueError("max_new_tasks must be positive")
    run_prefix = "acquisition-r12"
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
            raise ValueError("R12 resume path must use an acquisition-r12-* namespace")
        manifest_path = artifact_root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("R12 resume root is missing manifest.json")
        existing: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("run_revision") != "R12":
            raise ValueError("R12 rejects resume roots from R11, R10, R9, R8, or older revisions")

    configuration = _configured_runtime(
        load_experiment_configuration(project_root / R12_CONFIG, project_root=project_root),
        project_root / "benchmark/workspaces",
        project_root / "research/evidence/workspaces",
    )
    _validate_r12_harness_configuration(configuration)
    cases = _task_cases(configuration, ACQUISITION_TASK_IDS)
    settings = _settings_for_agent(
        coding_model=R12_EXPECTED_CODING_MODEL,
        abstraction_model=R12_EXPECTED_ABSTRACTION_MODEL,
    )
    _validate_preflight_configuration(
        configuration,
        settings,
        expected_coding_model=R12_EXPECTED_CODING_MODEL,
        expected_abstraction_model=R12_EXPECTED_ABSTRACTION_MODEL,
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
        objective_timeout_seconds=cast(float, configuration.config.objective_timeout_seconds),
    )
    configuration_hash = _r8_configuration_hash(configuration, settings, environments)
    manifest: dict[str, object] = {
        "gate": "GS-E003 / Gate A1",
        "phase": "acquisition",
        "run_revision": "R12",
        "namespace": R12_NAMESPACE,
        "task_ids": list(ACQUISITION_TASK_IDS),
        "status": "BLOCKED_GATE_A1_ACQUISITION_R12",
        "configuration_hash": configuration_hash,
        "coding_model": settings.openrouter_coding_model,
        "abstraction_model": settings.openrouter_abstraction_model,
        "prompt_version": configuration.model.prompt_version,
        "limits": configuration.config.limits.model_dump(mode="json"),
        "agent_timeout_seconds": R9_AGENT_TIMEOUT_SECONDS,
        "objective_timeout_seconds": R9_OBJECTIVE_TIMEOUT_SECONDS,
        "stopping_policy": R12_STOPPING_POLICY,
        "objective_mutation_check_policy": R12_OBJECTIVE_MUTATION_CHECK_POLICY,
        "workspace_line_ending_policy": R8_LINE_ENDING_POLICY,
        "persistence_session_policy": R8_PERSISTENCE_SESSION_POLICY,
        "persistence_retry_policy": R8_PERSISTENCE_RETRY_POLICY,
        "recovery_event_semantics": R12_RECOVERY_EVENT_SEMANTICS,
        "repository_mutation_evidence_policy": R9_REPOSITORY_MUTATION_EVIDENCE_POLICY,
        "command_argv_policy": R10_COMMAND_ARGV_POLICY,
        "revision_reason": R12_REVISION_REASON,
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
        if not isinstance(existing, dict) or existing.get("run_revision") != "R12":
            raise ValueError("R12 rejects resume roots from R11, R10, R9, R8, or older revisions")
        if existing.get("configuration_hash") != configuration_hash:
            raise ValueError("R12 resume configuration hash differs")
        manifest = existing
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
        run_id = f"GS-E003-A1-R12-{task_id}-{uuid.uuid4().hex}"
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
            revision="R12",
            condition="acquisition-r12",
            controller_type=R12ObjectiveController,
            objective_anchored_acquisition=True,
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
        task_records, completed_task_ids, ACQUISITION_TASK_IDS
    )
    status = status.replace("_R8", "_R12")
    manifest["completed_tasks"] = len(completed_task_ids)
    manifest["completed_task_ids"] = list(completed_task_ids)
    manifest["eligible_acquisition_tasks"] = eligible_acquisition_tasks
    manifest["acquisition_corpus_ready"] = corpus_ready
    manifest["status"] = status
    if status == "BLOCKED_GATE_A1_ACQUISITION_R12_NON_EVALUABLE":
        manifest["acquisition_reason"] = "zero_eligible_recovery_patterns"
    _write_json(manifest_path, manifest)
    return str(status), artifact_root


__all__ = [
    "R12ObjectiveController",
    "R12ObjectiveEvidenceError",
    "run_gate_a1_acquisition_r12",
]
