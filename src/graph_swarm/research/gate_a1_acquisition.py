"""Real, write-only Gate A1 acquisition for the frozen five-task pilot."""

# The Sprint 3 harness exposes the validated environment/workspace primitives
# as implementation helpers; this module deliberately reuses those exact
# boundaries without routing execution through the B0/O1 runner.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic_ai import AgentRunResult, ModelSettings
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

from experiments.sprint3 import (
    FrozenSWEsmithCase,
    FrozenSWEsmithObjective,
    IsolatedTaskEnvironment,
    _configured_runtime,
    _materialize_workspace,
    _verify_baselines,
    _verify_frozen_patches,
    load_frozen_swesmith_cases,
)
from graph_swarm.agent.coding_agent import create_coding_agent, run_coding_agent
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.pacing import ProviderRequestPacing
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.neo4j_repository import Neo4jRepository
from graph_swarm.integration.event_persistence import (
    MissingTrustedPlannedActionError,
    persist_agent_event_stream,
)
from graph_swarm.memory.recovery_abstraction import (
    FROZEN_RECOVERY_MODEL,
    abstract_and_persist_recovery_pattern,
)
from graph_swarm.memory.recovery_embeddings import (
    RecoveryPatternEmbedder,
    embed_and_persist_recovery_pattern,
)
from graph_swarm.research.benchmark_environments import load_benchmark_environment_policy
from graph_swarm.research.gate_a1 import (
    ACQUISITION_CONTRACT,
    _docker_image_status,
    assess_acquisition_environments,
    load_acquisition_identities,
)
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    build_task_prompt,
    load_experiment_configuration,
    load_task_cases,
)
from graph_swarm.settings import get_settings

ACQUISITION_TASK_IDS = tuple(f"GS-T{i:03d}" for i in range(1, 6))
ACQUISITION_NAMESPACE = "GS-E003/Gate-A1/acquisition"
ACQUISITION_CONDITION = "gate_a1_acquisition"


class GateA1AcquisitionPreflightError(RuntimeError):
    """Raised when the authoritative Gate A1 environment contract is unusable."""


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _settings_for_agent() -> Any:
    base = get_settings()
    return base.model_copy(
        update={
            "model_provider": "openrouter",
            "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY")
            or base.openrouter_api_key,
            "openrouter_model": os.environ.get("OPENROUTER_MODEL")
            or base.openrouter_model,
        }
    )


def _task_cases(configuration: Any, task_ids: Sequence[str]) -> list[BenchmarkTaskCase]:
    requested = set(task_ids)
    selected = {
        task_id: case
        for case in load_task_cases(
            configuration.task_manifest_path,
            problem_statements_path=configuration.task_problems_path,
        )
        if (task_id := case.task.id) in requested
    }
    if tuple(selected) != tuple(task_ids):
        raise RuntimeError("Gate A1 acquisition task manifest is incomplete or unordered")
    cases = [selected[task_id] for task_id in task_ids]
    if any(case.occurrence_index != 1 for case in cases):
        raise RuntimeError("Gate A1 acquisition requires occurrence-1 tasks only")
    return cases


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _required_contract_text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GateA1AcquisitionPreflightError(
            f"Gate A1 environment contract field {field!r} is missing or empty"
        )
    return value


def _load_contract_records(project_root: Path) -> dict[str, dict[str, object]]:
    contract_path = project_root / ACQUISITION_CONTRACT
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateA1AcquisitionPreflightError(
            f"could not read Gate A1 environment contract: {contract_path}"
        ) from error
    if not isinstance(raw, dict):
        raise GateA1AcquisitionPreflightError(
            "Gate A1 environment contract must contain an environments list"
        )
    raw_object = cast(dict[str, object], raw)
    if not isinstance(raw_object.get("environments"), list):
        raise GateA1AcquisitionPreflightError(
            "Gate A1 environment contract must contain an environments list"
        )
    records: dict[str, dict[str, object]] = {}
    for value in cast(list[object], raw_object["environments"]):
        if not isinstance(value, dict):
            raise GateA1AcquisitionPreflightError(
                "Gate A1 environment contract contains a non-object environment"
            )
        record = cast(dict[str, object], value)
        task_id = _required_contract_text(record, "task_id")
        if task_id in records:
            raise GateA1AcquisitionPreflightError(
                f"Gate A1 environment contract contains duplicate task {task_id}"
            )
        records[task_id] = record
    return records


def _manifest_instance_ids(manifest_path: Path, task_ids: Sequence[str]) -> dict[str, str]:
    wanted = set(task_ids)
    found: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record_raw = json.loads(line)
        if not isinstance(record_raw, dict):
            raise GateA1AcquisitionPreflightError("manifest contains a non-object record")
        record = cast(dict[str, object], record_raw)
        if record.get("task_id") not in wanted:
            continue
        task_id = str(record["task_id"])
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise GateA1AcquisitionPreflightError(
                f"manifest instance_id is missing for {task_id}"
            )
        if task_id in found:
            raise GateA1AcquisitionPreflightError(
                f"manifest contains duplicate task {task_id}"
            )
        found[task_id] = instance_id
    if set(found) != wanted:
        raise GateA1AcquisitionPreflightError(
            "manifest instance IDs are incomplete for the acquisition set"
        )
    return found


def resolve_gate_a1_environments(
    project_root: Path,
    *,
    task_ids: Sequence[str] = ACQUISITION_TASK_IDS,
) -> dict[str, IsolatedTaskEnvironment]:
    """Resolve current prepared runtimes from the authoritative Gate A1 contract."""
    requested = tuple(task_ids)
    if requested != ACQUISITION_TASK_IDS:
        raise GateA1AcquisitionPreflightError(
            "acquisition environment resolution is frozen to GS-T001 through GS-T005"
        )
    statuses = assess_acquisition_environments(project_root, task_ids=requested)
    status_by_task = {status.task_id: status for status in statuses}
    if tuple(status_by_task) != requested or len(statuses) != len(requested):
        raise GateA1AcquisitionPreflightError(
            "Gate A1 readiness did not return exactly the five acquisition environments"
        )
    blocked = {
        status.task_id: status.blockers for status in statuses if not status.ready
    }
    if blocked:
        raise GateA1AcquisitionPreflightError(f"Gate A1 environment readiness blocked: {blocked}")

    identities = load_acquisition_identities(project_root, task_ids=requested)
    contract = _load_contract_records(project_root)
    if set(contract) != set(requested):
        raise GateA1AcquisitionPreflightError(
            "Gate A1 environment contract must contain exactly T001-T005"
        )

    environments: dict[str, IsolatedTaskEnvironment] = {}
    for identity in identities:
        record = contract[identity.task_id]
        repository = _required_contract_text(record, "repository")
        upstream_image = _required_contract_text(record, "upstream_image")
        upstream_digest = _required_contract_text(record, "upstream_digest")
        prepared_image = _required_contract_text(record, "prepared_image")
        fingerprint = _required_contract_text(record, "environment_fingerprint")
        marker_value = _required_contract_text(record, "validation_marker")
        workspace_value = _required_contract_text(record, "workspace")
        if repository != identity.repository or upstream_image != identity.upstream_image:
            raise GateA1AcquisitionPreflightError(
                f"Gate A1 environment identity mismatch for {identity.task_id}"
            )

        marker = _project_path(project_root, marker_value).resolve()
        workspace = _project_path(project_root, workspace_value).resolve()
        try:
            marker_raw = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise GateA1AcquisitionPreflightError(
                f"Gate A1 validation marker is missing or invalid for {identity.task_id}"
            ) from error
        if not isinstance(marker_raw, dict):
            raise GateA1AcquisitionPreflightError(
                f"Gate A1 validation marker is not an object for {identity.task_id}"
            )
        marker_metadata = cast(dict[str, object], marker_raw)
        marker_checks = {
            "task_id": identity.task_id,
            "validated": True,
            "runtime_type": "docker",
            "environment_fingerprint": fingerprint,
            "container_image": prepared_image,
            "base_container_image": upstream_image,
            "base_image_digest": upstream_digest,
        }
        for field, expected in marker_checks.items():
            if marker_metadata.get(field) != expected:
                raise GateA1AcquisitionPreflightError(
                    f"Gate A1 validation marker mismatch for {identity.task_id}: {field}"
                )
        if marker.parent.name != fingerprint:
            raise GateA1AcquisitionPreflightError(
                f"Gate A1 validation marker path fingerprint mismatch for {identity.task_id}"
            )
        if not workspace.is_dir():
            raise GateA1AcquisitionPreflightError(
                f"Gate A1 workspace contract is missing for {identity.task_id}: {workspace}"
            )
        image_ready, image_error = _docker_image_status(prepared_image)
        if not image_ready:
            raise GateA1AcquisitionPreflightError(
                f"prepared image is unavailable for {identity.task_id}: {image_error}"
            )
        python_executable = _required_contract_text(marker_metadata, "python_executable")
        python_version = _required_contract_text(marker_metadata, "python_version")
        container_python = _required_contract_text(
            marker_metadata, "container_python_executable"
        )
        environments[identity.task_id] = IsolatedTaskEnvironment(
            task_id=identity.task_id,
            python_executable=Path(python_executable),
            validation_marker=marker,
            python_version=python_version,
            environment_fingerprint=fingerprint,
            runtime_type="docker",
            container_image=prepared_image,
            container_python_executable=container_python,
            base_container_image=upstream_image,
            benchmark_policy_path=(
                project_root / "configs/research/gate_a1_benchmark_environments.toml"
            ),
            benchmark_manifest_path=project_root / "benchmark/manifests/pilot.jsonl",
        )
    return environments


def _validate_preflight_configuration(configuration: Any, settings: Any) -> None:
    if configuration.model.provider != "openrouter" or settings.model_provider != "openrouter":
        raise GateA1AcquisitionPreflightError("Gate A1 coding provider must be OpenRouter")
    if not settings.openrouter_api_key or not settings.openrouter_api_key.strip():
        raise GateA1AcquisitionPreflightError("OPENROUTER_API_KEY is missing")
    if not settings.openrouter_model or not settings.openrouter_model.strip():
        raise GateA1AcquisitionPreflightError("OPENROUTER_MODEL is missing")


def run_gate_a1_acquisition_preflight(project_root: Path) -> tuple[str, Path]:
    """Validate Gate A1 execution prerequisites without objective/provider execution."""
    project_root = project_root.expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_root = (
        project_root
        / "research/evidence/results/GS-E003/gate_a1"
        / f"acquisition-preflight-{stamp}"
    )
    manifest: dict[str, object] = {
        "gate": "GS-E003 / Gate A1",
        "phase": "acquisition_preflight",
        "status": "BLOCKED_GATE_A1_ACQUISITION_PREFLIGHT",
        "task_ids": list(ACQUISITION_TASK_IDS),
        "tasks_attempted": 0,
        "provider_calls": 0,
        "retrieval_executed": False,
        "transfer_tasks_executed": 0,
        "behavioral_evaluation_executed": False,
        "objective_data_loaded": False,
        "errors": [],
    }
    try:
        baseline_root = project_root / "benchmark/workspaces"
        configuration = _configured_runtime(
            load_experiment_configuration(
                project_root / "configs/experiments/rollout_3a_pilot.yaml",
                project_root=project_root,
            ),
            baseline_root,
            project_root / "research/evidence/workspaces",
        )
        cases = _task_cases(configuration, ACQUISITION_TASK_IDS)
        environments = resolve_gate_a1_environments(project_root)
        if tuple(case.task.id for case in cases) != ACQUISITION_TASK_IDS:
            raise GateA1AcquisitionPreflightError("acquisition chronology is not canonical")
        settings = _settings_for_agent()
        _validate_preflight_configuration(configuration, settings)
        repository = Neo4jRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        try:
            repository.verify_connectivity()
            repository.ensure_recovery_pattern_vector_index()
        finally:
            repository.close()
        manifest["tasks"] = [
            {
                "task_id": case.task.id,
                "chronological_index": case.task.chronological_index,
                "runtime_type": environments[case.task.id].runtime_type,
                "prepared_image": environments[case.task.id].container_image,
                "environment_fingerprint": environments[case.task.id].environment_fingerprint,
                "validation_marker": str(environments[case.task.id].validation_marker),
                "status": "ready",
            }
            for case in cases
        ]
        manifest["status"] = "READY_FOR_GATE_A1_ACQUISITION_EXECUTION"
    except Exception as error:
        manifest["errors"] = [{"type": type(error).__name__, "message": str(error)}]
    _write_json(artifact_root / "manifest.json", manifest)
    return str(manifest["status"]), artifact_root


def _event_trace(dependencies: AgentDependencies) -> list[dict[str, object]]:
    return [event.model_dump(mode="json") for event in dependencies.events]


def _trusted_actions(dependencies: AgentDependencies) -> list[dict[str, object]]:
    return [action.model_dump(mode="json") for action in dependencies.planned_actions.values()]


def _result_messages(result: AgentRunResult[str] | None) -> list[object]:
    if result is None:
        return []
    try:
        return cast(list[object], json.loads(result.all_messages_json()))
    except (TypeError, ValueError):
        return []


def _base_task_artifact(
    *,
    task: Task,
    run: Run,
    environment: IsolatedTaskEnvironment,
    model: str | None,
    model_settings: Mapping[str, object],
    prompt_version: str,
    workspace: Path,
) -> dict[str, object]:
    runtime = environment.agent_execution_runtime()
    return {
        "namespace": ACQUISITION_NAMESPACE,
        "gate": "GS-E003 / Gate A1",
        "phase": "acquisition",
        "task_id": task.id,
        "chronological_index": task.chronological_index,
        "run_id": run.id,
        "started_at": run.started_at.isoformat(),
        "workspace": str(workspace),
        "runtime_type": runtime.runtime_type,
        "container_image": environment.container_image,
        "container_python_executable": environment.container_python_executable,
        "environment_fingerprint": environment.environment_fingerprint,
        "model": model,
        "model_settings": dict(model_settings),
        "prompt_version": prompt_version,
        "prompt_policy": {
            "graph_swarm_advice": False,
            "family_id": False,
            "gold_patch": False,
            "evaluator_metadata": False,
            "future_task_metadata": False,
        },
        "retrieval_performed": False,
        "transfer_performed": False,
        "behavioral_evaluation_performed": False,
    }


def _preflight(
    *,
    project_root: Path,
    baseline_root: Path,
    execution_root: Path,
    configuration: Any,
    cases: Sequence[BenchmarkTaskCase],
) -> tuple[dict[str, IsolatedTaskEnvironment], dict[str, FrozenSWEsmithCase]]:
    environments = resolve_gate_a1_environments(project_root)
    policy = load_benchmark_environment_policy(
        project_root / "configs/research/gate_a1_benchmark_environments.toml",
        expected_task_order=ACQUISITION_TASK_IDS,
    )
    _verify_baselines(cases, baseline_root)
    instance_ids = _manifest_instance_ids(
        configuration.task_manifest_path, ACQUISITION_TASK_IDS
    )
    # The frozen objective rows are prerequisite benchmark data, not task
    # trajectory data. They are read-only and are never passed to the agent.
    frozen_by_instance = load_frozen_swesmith_cases(
        tuple(instance_ids.values()), allow_network=True
    )
    frozen_cases = {
        case.task.id: frozen_by_instance[instance_ids[case.task.id].lower()] for case in cases
    }
    _verify_frozen_patches(cases, baseline_root, frozen_cases)
    if tuple(policy.task_order) != ACQUISITION_TASK_IDS:
        raise GateA1AcquisitionPreflightError("Gate A1 benchmark policy chronology changed")
    objective = FrozenSWEsmithObjective(frozen_cases, environments)
    for case in cases:
        workspace = _materialize_workspace(
            source_root=baseline_root,
            execution_root=execution_root,
            frozen_cases=frozen_cases,
            condition="gate_a1_preflight",
            task=case.task,
        )
        objective.preflight(case.task, workspace)
    return environments, frozen_cases


def _run_task(
    *,
    baseline_root: Path,
    execution_root: Path,
    artifact_root: Path,
    configuration: Any,
    case: BenchmarkTaskCase,
    environment: IsolatedTaskEnvironment,
    objective: FrozenSWEsmithObjective,
    repository: Neo4jRepository,
    settings: Any,
    pacing: ProviderRequestPacing,
    embedder: RecoveryPatternEmbedder | None,
) -> tuple[dict[str, object], RecoveryPatternEmbedder | None, bool]:
    task = case.task
    run_id = f"GS-E003-A1-{task.id}-{uuid.uuid4().hex}"
    run = Run(id=run_id, task_id=task.id, started_at=datetime.now(UTC))
    workspace = _materialize_workspace(
        source_root=baseline_root,
        execution_root=execution_root,
        frozen_cases=objective.cases,
        condition=ACQUISITION_CONDITION,
        task=task,
    )
    run_dir = artifact_root / "GS-E003" / "gate_a1" / "acquisition" / task.id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    step_database = run_dir / "steps.sqlite"
    artifact = _base_task_artifact(
        task=task,
        run=run,
        environment=environment,
        model=settings.openrouter_model,
        model_settings=configuration.model.settings,
        prompt_version=configuration.model.prompt_version,
        workspace=workspace,
    )
    environment_context = EnvironmentContext(
        id=f"{run_id}-environment",
        repository=task.repository,
        runtime="docker",
        versions={"python": environment.python_version or "unknown"},
        markers={
            "gate": "GS-E003/Gate-A1",
            "acquisition_namespace": run_id,
            "environment_fingerprint": environment.environment_fingerprint or "unknown",
            "prepared_container_image": environment.container_image or "unknown",
            "memory_write_only": "true",
            "retrieval_performed": "false",
        },
    )
    repository.save_task(task)
    repository.save_run(run)
    repository.save_environment(environment_context)
    dependencies = AgentDependencies(
        workspace_root=workspace,
        run_id=run_id,
        task_id=task.id,
        task=None,
        environment=None,
        advisory_service=None,
        artifact_writer=None,
        execution_runtime=environment.agent_execution_runtime(),
    )
    result: AgentRunResult[str] | None = None
    error: Exception | None = None
    started_counter = time.perf_counter()
    try:
        step_persistence = StepPersistence(
            store=SqliteStepStore(database=step_database),
            agent_name="graph_swarm_coding_agent",
            run_id=run_id,
            metadata={
                "gate": "GS-E003 / Gate A1",
                "phase": "acquisition",
                "task_id": task.id,
            },
        )
        agent = create_coding_agent(settings, capabilities=[step_persistence])
        result = run_coding_agent(
            agent,
            settings,
            dependencies,
            build_task_prompt(task),
            max_actions=configuration.config.limits.max_actions,
            max_requests=configuration.config.limits.max_requests,
            timeout_seconds=configuration.config.limits.timeout_seconds,
            model_settings=cast(ModelSettings, configuration.model.settings),
            request_pacing=pacing,
        )
    except Exception as caught:  # preserve real provider/runtime failures in evidence
        error = caught

    objective_result = False
    objective_error: Exception | None = None
    try:
        objective_result = objective(task, workspace)
    except Exception as caught:
        objective_error = caught
        error = error or caught

    persistence_error: Exception | None = None
    recovery_lineage: dict[str, object] | None = None
    pattern_records: list[dict[str, object]] = []
    failure_records: list[dict[str, object]] = []
    trusted = True
    try:
        persisted = persist_agent_event_stream(
            repository,
            dependencies.events,
            task,
            run,
            environment_context,
            planned_actions=dependencies.planned_actions,
            require_trusted_planned_actions=True,
        )
        for event in dependencies.events:
            if not event.result.success:
                failure_records.append(
                    {
                        "action_id": event.action_id,
                        "tool": event.result.tool_name,
                        "error": event.result.error,
                    }
                )
        if persisted is not None:
            failure, resolution, outcome = persisted
            recovery_lineage = {
                "failure_id": failure.id,
                "resolution_id": resolution.id,
                "outcome_id": outcome.id,
            }
            lineage = repository.get_recovery_evidence(failure.id)
            pattern = abstract_and_persist_recovery_pattern(lineage, repository, settings)
            if embedder is None:
                embedder = RecoveryPatternEmbedder()
            embedded = embed_and_persist_recovery_pattern(pattern, repository, embedder)
            pattern_records.append(
                {
                    "pattern_id": embedded.id,
                    "verification_status": embedded.verification_status.value,
                    "embedding_dimension": len(embedded.embedding or []),
                    "embedding_normalized": True,
                }
            )
    except MissingTrustedPlannedActionError as caught:
        trusted = False
        persistence_error = caught
    except Exception as caught:  # preserve persistence/abstraction/embedding failures
        persistence_error = caught

    if persistence_error is not None:
        error = error or persistence_error
    status = "acquired"
    if not trusted:
        status = "blocked_missing_trusted_planned_action_provenance"
    elif objective_error is not None:
        status = "blocked_objective_evaluator_error"
    elif error is not None:
        status = "blocked_runtime_or_persistence_error"
    elif recovery_lineage is None:
        status = "acquired_zero_valid_recovery_lineages"
    elif not pattern_records:
        status = "blocked_pattern_materialization"

    artifact.update(
        {
            "status": status,
            "duration_seconds": time.perf_counter() - started_counter,
            "agent_error": None if error is None else type(error).__name__,
            "agent_error_message": None if error is None else str(error),
            "objective": {
                "task_success": objective_result,
                "objective_error": (
                    None if objective_error is None else type(objective_error).__name__
                ),
                "observation": (
                    dataclasses.asdict(objective.observations[-1])
                    if objective.observations and objective.observations[-1].task_id == task.id
                    else None
                ),
            },
            "counts": {
                "events": len(dependencies.events),
                "trusted_planned_actions": len(dependencies.planned_actions),
                "failures": len(failure_records),
                "recoveries": 0 if recovery_lineage is None else 1,
                "lineages": 0 if recovery_lineage is None else 1,
                "patterns": len(pattern_records),
            },
            "failures": failure_records,
            "recovery_lineage": recovery_lineage,
            "patterns": pattern_records,
            "persistence_error": (
                None if persistence_error is None else type(persistence_error).__name__
            ),
            "trusted_planned_action_provenance": trusted,
            "events": _event_trace(dependencies),
            "trusted_planned_actions": _trusted_actions(dependencies),
            "messages": _result_messages(result),
            "agent_output": None if result is None else result.output,
            "usage": None if result is None else dataclasses.asdict(result.usage),
            "step_database": str(step_database),
        }
    )
    _write_json(run_dir / "acquisition.json", artifact)
    return artifact, embedder, status.startswith("acquired")


def run_gate_a1_acquisition(project_root: Path) -> tuple[str, Path]:
    """Run exactly T001-T005 acquisition and return final status plus artifact root."""
    project_root = project_root.expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_root = (
        project_root / "research/evidence/results/GS-E003/gate_a1" / f"acquisition-{stamp}"
    )
    manifest: dict[str, object] = {
        "gate": "GS-E003 / Gate A1",
        "phase": "acquisition",
        "namespace": ACQUISITION_NAMESPACE,
        "task_ids": list(ACQUISITION_TASK_IDS),
        "transfer_tasks_executed": [],
        "retrieval_executed": False,
        "behavioral_evaluation_executed": False,
        "status": "BLOCKED_GATE_A1_ACQUISITION",
        "tasks": [],
        "errors": [],
    }
    try:
        baseline_root = project_root / "benchmark/workspaces"
        execution_root = project_root / "research/evidence/workspaces"
        configuration = _configured_runtime(
            load_experiment_configuration(
                project_root / "configs/experiments/rollout_3a_pilot.yaml",
                project_root=project_root,
            ),
            baseline_root,
            execution_root,
        )
        cases = _task_cases(configuration, ACQUISITION_TASK_IDS)
        environments, frozen_cases = _preflight(
            project_root=project_root,
            baseline_root=baseline_root,
            execution_root=execution_root,
            configuration=configuration,
            cases=cases,
        )
        settings = _settings_for_agent()
        repository = Neo4jRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        previous_api_key = os.environ.get("OPENROUTER_API_KEY")
        previous_model = os.environ.get("OPENROUTER_MODEL")
        if settings.openrouter_api_key:
            os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key
        os.environ["OPENROUTER_MODEL"] = FROZEN_RECOVERY_MODEL
        try:
            repository.verify_connectivity()
            repository.ensure_recovery_pattern_vector_index()
            objective = FrozenSWEsmithObjective(frozen_cases, environments)
            pacing = ProviderRequestPacing()
            embedder: RecoveryPatternEmbedder | None = None
            task_artifacts: list[dict[str, object]] = []
            for case in cases:
                task_artifact, embedder, _ = _run_task(
                    baseline_root=baseline_root,
                    execution_root=execution_root,
                    artifact_root=artifact_root,
                    configuration=configuration,
                    case=case,
                    environment=environments[case.task.id],
                    objective=objective,
                    repository=repository,
                    settings=settings,
                    pacing=pacing,
                    embedder=embedder,
                )
                task_artifacts.append(task_artifact)
            manifest["tasks"] = task_artifacts
            manifest["completed_tasks"] = len(task_artifacts)
            def count_value(task_artifact: dict[str, object], key: str) -> int:
                counts = cast(dict[str, object], task_artifact["counts"])
                value = counts[key]
                if not isinstance(value, (int, float, str)):
                    raise TypeError(f"invalid acquisition count for {key}: {value!r}")
                return int(value)

            manifest["counts"] = {
                "failures": sum(count_value(c, "failures") for c in task_artifacts),
                "recoveries": sum(count_value(c, "recoveries") for c in task_artifacts),
                "lineages": sum(count_value(c, "lineages") for c in task_artifacts),
                "patterns": sum(count_value(c, "patterns") for c in task_artifacts),
            }
            blocked = [
                str(c["task_id"])
                for c in task_artifacts
                if not str(c["status"]).startswith("acquired")
            ]
            if len(task_artifacts) == len(ACQUISITION_TASK_IDS) and not blocked:
                manifest["status"] = "READY_FOR_GATE_A1_RETRIEVAL_EVALUATION"
            else:
                manifest["errors"] = [{"blocked_tasks": blocked}]
        finally:
            repository.close()
            if previous_api_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = previous_api_key
            if previous_model is None:
                os.environ.pop("OPENROUTER_MODEL", None)
            else:
                os.environ["OPENROUTER_MODEL"] = previous_model
    except Exception as error:
        manifest["errors"] = [{"type": type(error).__name__, "message": str(error)}]
    _write_json(artifact_root / "manifest.json", manifest)
    return str(manifest["status"]), artifact_root


__all__ = ["run_gate_a1_acquisition"]
