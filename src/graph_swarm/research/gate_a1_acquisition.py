"""Real, write-only Gate A1 acquisition for the frozen five-task pilot."""

# The Sprint 3 harness exposes the validated environment/workspace primitives
# as implementation helpers; this module deliberately reuses those exact
# boundaries without routing execution through the B0/O1 runner.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
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
ACQUISITION_R2_NAMESPACE = "GS-E003/Gate-A1/acquisition-r2"
ACQUISITION_CONDITION = "gate_a1_acquisition"
ACQUISITION_R2_CONDITION = "gate_a1_acquisition_r2"
ACQUISITION_R2_CONFIG = "configs/experiments/gate_a1_acquisition_r2.yaml"
ACQUISITION_R3_NAMESPACE = "GS-E003/Gate-A1/acquisition-r3"
ACQUISITION_R3_CONDITION = "gate_a1_acquisition_r3"
ACQUISITION_R3_CONFIG = "configs/experiments/gate_a1_acquisition_r3.yaml"
ACQUISITION_R4_NAMESPACE = "GS-E003/Gate-A1/acquisition-r4"
ACQUISITION_R4_CONDITION = "gate_a1_acquisition_r4"
ACQUISITION_R4_CONFIG = "configs/experiments/gate_a1_acquisition_r4.yaml"
ACQUISITION_R5_NAMESPACE = "GS-E003/Gate-A1/acquisition-r5"
ACQUISITION_R5_CONDITION = "gate_a1_acquisition_r5"
ACQUISITION_R5_CONFIG = "configs/experiments/gate_a1_acquisition_r5.yaml"
FROZEN_CODING_MODEL = "qwen/qwen3-coder:free"
FROZEN_R3_CODING_MODEL = "cohere/north-mini-code:free"
FROZEN_R4_CODING_MODEL = "cohere/north-mini-code:free"
R5_EXPECTED_CODING_MODEL = "nex-agi/nex-n2.5-pro:free"
R5_EXPECTED_ABSTRACTION_MODEL = "cohere/north-mini-code:free"


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


def _settings_for_agent(
    *,
    coding_model: str | None = None,
    abstraction_model: str | None = None,
) -> Any:
    """Resolve both OpenRouter roles through the typed Settings object.

    The optional overrides are used only by historical R2-R4 replay paths,
    whose model identities are part of their frozen provenance. R5 passes no
    overrides and therefore requires both role-specific environment settings.
    """
    cache_clear = getattr(get_settings, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    base = get_settings()
    updates: dict[str, object] = {"model_provider": "openrouter"}
    if coding_model is not None:
        updates["openrouter_coding_model"] = coding_model
    if abstraction_model is not None:
        updates["openrouter_abstraction_model"] = abstraction_model
    return base.model_copy(
        update=updates,
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


def _validate_preflight_configuration(
    configuration: Any,
    settings: Any,
    *,
    expected_coding_model: str | None = FROZEN_CODING_MODEL,
    expected_abstraction_model: str | None = FROZEN_RECOVERY_MODEL,
) -> None:
    if configuration.model.provider != "openrouter" or settings.model_provider != "openrouter":
        raise GateA1AcquisitionPreflightError("Gate A1 coding provider must be OpenRouter")
    if not settings.openrouter_api_key or not settings.openrouter_api_key.strip():
        raise GateA1AcquisitionPreflightError("OPENROUTER_API_KEY is missing")
    coding_model = getattr(settings, "openrouter_coding_model", None)
    abstraction_model = getattr(settings, "openrouter_abstraction_model", None)
    if not isinstance(coding_model, str) or not coding_model.strip():
        raise GateA1AcquisitionPreflightError("OPENROUTER_CODING_MODEL is missing")
    if not isinstance(abstraction_model, str) or not abstraction_model.strip():
        raise GateA1AcquisitionPreflightError("OPENROUTER_ABSTRACTION_MODEL is missing")
    if expected_coding_model is not None and coding_model != expected_coding_model:
        raise GateA1AcquisitionPreflightError(
            "Gate A1 coding model must be "
            f"{expected_coding_model!r}"
        )
    if expected_abstraction_model is not None and abstraction_model != expected_abstraction_model:
        raise GateA1AcquisitionPreflightError(
            "Gate A1 abstraction model must be "
            f"{expected_abstraction_model!r}"
        )


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
    namespace: str = ACQUISITION_NAMESPACE,
) -> dict[str, object]:
    runtime = environment.agent_execution_runtime()
    return {
        "namespace": namespace,
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
    run_id: str | None = None,
    namespace: str = ACQUISITION_NAMESPACE,
    condition: str = ACQUISITION_CONDITION,
    artifact_subdirectory: str = "acquisition",
) -> tuple[dict[str, object], RecoveryPatternEmbedder | None, bool]:
    task = case.task
    run_id = run_id or f"GS-E003-A1-{task.id}-{uuid.uuid4().hex}"
    run = Run(id=run_id, task_id=task.id, started_at=datetime.now(UTC))
    workspace = _materialize_workspace(
        source_root=baseline_root,
        execution_root=execution_root,
        frozen_cases=objective.cases,
        condition=condition,
        task=task,
    )
    run_dir = artifact_root / "GS-E003" / "gate_a1" / artifact_subdirectory / task.id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    step_database = run_dir / "steps.sqlite"
    artifact = _base_task_artifact(
        task=task,
        run=run,
        environment=environment,
        model=getattr(settings, "openrouter_coding_model", None),
        model_settings=configuration.model.settings,
        prompt_version=configuration.model.prompt_version,
        workspace=workspace,
        namespace=namespace,
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


def _r2_configuration_hash(
    configuration: Any,
    settings: Any,
    environments: Mapping[str, IsolatedTaskEnvironment],
    *,
    max_new_tasks: int | None = None,
) -> str:
    """Hash the immutable model roles, prompt, budget, and runtime boundary.

    ``max_new_tasks`` is deliberately an orchestration control, not part of
    the task-level configuration identity.
    """
    del max_new_tasks
    coding_model = getattr(settings, "openrouter_coding_model", None)
    if not isinstance(coding_model, str) or not coding_model.strip():
        # This compatibility branch only supports historical test/replay
        # inputs that predate role-specific Settings fields. It is not used by
        # the R5 runtime path.
        coding_model = getattr(settings, "openrouter_model", None)
    abstraction_model = getattr(settings, "openrouter_abstraction_model", None)
    if not isinstance(abstraction_model, str) or not abstraction_model.strip():
        abstraction_model = FROZEN_RECOVERY_MODEL
    if not isinstance(coding_model, str) or not coding_model.strip():
        raise GateA1AcquisitionPreflightError(
            "OPENROUTER_CODING_MODEL is missing from the resolved run settings"
        )
    payload = {
        "task_ids": list(ACQUISITION_TASK_IDS),
        "coding_model": coding_model,
        "abstraction_model": abstraction_model,
        "prompt_version": configuration.model.prompt_version,
        "model_settings": dict(configuration.model.settings),
        "limits": {
            "max_actions": configuration.config.limits.max_actions,
            "max_requests": configuration.config.limits.max_requests,
            "timeout_seconds": configuration.config.limits.timeout_seconds,
        },
        "environments": [
            {
                "task_id": task_id,
                "environment_fingerprint": environments[task_id].environment_fingerprint,
                "container_image": environments[task_id].container_image,
            }
            for task_id in ACQUISITION_TASK_IDS
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _r2_task_marker(
    artifact_root: Path,
    task_id: str,
    marker_name: str,
) -> Path:
    return artifact_root / "tasks" / task_id / marker_name


def _r2_resume_action(artifact_root: Path, task_id: str) -> str | None:
    """Return the durable resume decision for one task, if it already exists."""
    if _r2_task_marker(artifact_root, task_id, "completed.json").is_file():
        return "skipped_completed"
    if _r2_task_marker(artifact_root, task_id, "started.json").is_file():
        return "skipped_started_no_rerun"
    return None


def _r2_task_plan(
    artifact_root: Path,
    *,
    max_new_tasks: int | None,
) -> tuple[tuple[str, str], ...]:
    """Plan canonical R2 task handling without starting any provider call."""
    if max_new_tasks is not None and max_new_tasks <= 0:
        raise GateA1AcquisitionPreflightError("max_new_tasks must be positive")

    started_count = 0
    plan: list[tuple[str, str]] = []
    for task_id in ACQUISITION_TASK_IDS:
        resume_action = _r2_resume_action(artifact_root, task_id)
        if resume_action is not None:
            plan.append((task_id, resume_action))
            continue
        if max_new_tasks is not None and started_count >= max_new_tasks:
            break
        plan.append((task_id, "start"))
        started_count += 1
    return tuple(plan)


def _first_never_started_task(artifact_root: Path) -> str | None:
    """Return the first canonical task without a durable started marker."""
    for task_id in ACQUISITION_TASK_IDS:
        if not _r2_task_marker(artifact_root, task_id, "started.json").is_file():
            return task_id
    return None


def _completed_task_count(artifact_root: Path) -> int:
    """Count canonical tasks with durable completed markers."""
    return sum(
        _r2_task_marker(artifact_root, task_id, "completed.json").is_file()
        for task_id in ACQUISITION_TASK_IDS
    )


def _reconcile_execution_boundary(
    artifact_root: Path,
    manifest: dict[str, object],
    *,
    boundary_hit: bool,
) -> bool:
    """Reconcile a pause report with durable markers before final status write."""
    next_task_id = _first_never_started_task(artifact_root)
    if not boundary_hit or next_task_id is None:
        manifest.pop("next_task_id", None)
        return False
    manifest["next_task_id"] = next_task_id
    return True


def _validate_revision_manifest(
    manifest: Mapping[str, object],
    *,
    configuration_hash: str,
    revision: str,
    namespace: str,
    coding_model: str | None = None,
    abstraction_model: str | None = None,
) -> None:
    if manifest.get("run_revision") != revision:
        raise GateA1AcquisitionPreflightError(
            f"resume requires a Gate A1 Acquisition {revision} manifest"
        )
    if manifest.get("namespace") != namespace:
        raise GateA1AcquisitionPreflightError(
            f"resume requires the fresh Gate A1 Acquisition {revision} namespace"
        )
    if coding_model is not None and manifest.get("coding_model") != coding_model:
        raise GateA1AcquisitionPreflightError(
            f"{revision} resume coding model configuration does not match the "
            "immutable run configuration"
        )
    if abstraction_model is not None and manifest.get("abstraction_model") != abstraction_model:
        raise GateA1AcquisitionPreflightError(
            f"{revision} resume abstraction model configuration does not match the "
            "immutable run configuration"
        )
    if manifest.get("configuration_hash") != configuration_hash:
        raise GateA1AcquisitionPreflightError(
            f"{revision} resume configuration hash differs from the frozen initial attempt"
        )
    task_ids = manifest.get("task_ids")
    task_id_values = cast(list[object], task_ids) if isinstance(task_ids, list) else None
    if task_id_values is None or not all(isinstance(item, str) for item in task_id_values):
        raise GateA1AcquisitionPreflightError(
            f"{revision} resume task chronology must remain T001 through T005"
        )
    if tuple(cast(list[str], task_id_values)) != ACQUISITION_TASK_IDS:
        raise GateA1AcquisitionPreflightError(
            f"{revision} resume task chronology must remain T001 through T005"
        )


def _validate_r2_manifest(
    manifest: Mapping[str, object],
    *,
    configuration_hash: str,
) -> None:
    _validate_revision_manifest(
        manifest,
        configuration_hash=configuration_hash,
        revision="R2",
        namespace=ACQUISITION_R2_NAMESPACE,
    )


def _validate_r3_manifest(
    manifest: Mapping[str, object],
    *,
    configuration_hash: str,
) -> None:
    _validate_revision_manifest(
        manifest,
        configuration_hash=configuration_hash,
        revision="R3",
        namespace=ACQUISITION_R3_NAMESPACE,
    )


def _validate_r4_manifest(
    manifest: Mapping[str, object],
    *,
    configuration_hash: str,
) -> None:
    _validate_revision_manifest(
        manifest,
        configuration_hash=configuration_hash,
        revision="R4",
        namespace=ACQUISITION_R4_NAMESPACE,
    )


def _validate_r5_manifest(
    manifest: Mapping[str, object],
    *,
    configuration_hash: str,
    coding_model: str,
    abstraction_model: str,
) -> None:
    _validate_revision_manifest(
        manifest,
        configuration_hash=configuration_hash,
        revision="R5",
        namespace=ACQUISITION_R5_NAMESPACE,
        coding_model=coding_model,
        abstraction_model=abstraction_model,
    )


def run_gate_a1_acquisition_r2(
    project_root: Path,
    *,
    resume_root: Path | None = None,
    max_new_tasks: int | None = None,
    _revision: str = "R2",
) -> tuple[str, Path]:
    """Run or explicitly resume a uniform, immutable Gate A1 revision.

    A durable started marker is written before each real agent call.  A task
    with either a started or completed marker is never automatically rerun.
    The revision namespace and configuration hash prevent importing A1 state or
    resuming with a different model, prompt, budget, or environment.
    """
    if _revision not in {"R2", "R3", "R4", "R5"}:
        raise ValueError(f"unsupported Gate A1 acquisition revision: {_revision}")
    if max_new_tasks is not None and max_new_tasks <= 0:
        raise GateA1AcquisitionPreflightError("max_new_tasks must be positive")
    manifest_validator: Callable[..., None]
    if _revision == "R2":
        namespace = ACQUISITION_R2_NAMESPACE
        condition = ACQUISITION_R2_CONDITION
        configuration_path = ACQUISITION_R2_CONFIG
        expected_coding_model = FROZEN_CODING_MODEL
        expected_abstraction_model = FROZEN_RECOVERY_MODEL
        manifest_validator = _validate_r2_manifest
    elif _revision == "R3":
        namespace = ACQUISITION_R3_NAMESPACE
        condition = ACQUISITION_R3_CONDITION
        configuration_path = ACQUISITION_R3_CONFIG
        expected_coding_model = FROZEN_R3_CODING_MODEL
        expected_abstraction_model = FROZEN_RECOVERY_MODEL
        manifest_validator = _validate_r3_manifest
    else:
        if _revision == "R4":
            namespace = ACQUISITION_R4_NAMESPACE
            condition = ACQUISITION_R4_CONDITION
            configuration_path = ACQUISITION_R4_CONFIG
            expected_coding_model = FROZEN_R4_CODING_MODEL
            expected_abstraction_model = FROZEN_RECOVERY_MODEL
            manifest_validator = _validate_r4_manifest
        else:
            namespace = ACQUISITION_R5_NAMESPACE
            condition = ACQUISITION_R5_CONDITION
            configuration_path = ACQUISITION_R5_CONFIG
            expected_coding_model = R5_EXPECTED_CODING_MODEL
            expected_abstraction_model = R5_EXPECTED_ABSTRACTION_MODEL
            manifest_validator = _validate_r5_manifest
    run_prefix = f"acquisition-{_revision.lower()}"
    task_artifact_subdirectory = run_prefix
    ready_to_resume_status = f"READY_TO_RESUME_GATE_A1_ACQUISITION_{_revision}"
    blocked_status = f"BLOCKED_GATE_A1_ACQUISITION_{_revision}"
    project_root = project_root.expanduser().resolve()
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
            raise GateA1AcquisitionPreflightError(
                f"{_revision} resume path must use an {run_prefix}-* namespace"
            )

    manifest: dict[str, object] = {
        "gate": "GS-E003 / Gate A1",
        "phase": "acquisition",
        "run_revision": _revision,
        "namespace": namespace,
        "task_ids": list(ACQUISITION_TASK_IDS),
        "transfer_tasks_executed": [],
        "retrieval_executed": False,
        "behavioral_evaluation_executed": False,
        "status": blocked_status,
        "max_new_tasks": max_new_tasks,
        "new_tasks_started": 0,
        "tasks": [],
        "errors": [],
    }
    repository: Neo4jRepository | None = None
    resume_validation_failed = False
    try:
        baseline_root = project_root / "benchmark/workspaces"
        execution_root = project_root / "research/evidence/workspaces"
        configuration = _configured_runtime(
            load_experiment_configuration(
                project_root / configuration_path,
                project_root=project_root,
            ),
            baseline_root,
            execution_root,
        )
        cases = _task_cases(configuration, ACQUISITION_TASK_IDS)
        if _revision == "R5":
            settings = _settings_for_agent()
        else:
            settings = _settings_for_agent(
                coding_model=configuration.model.model,
                abstraction_model=FROZEN_RECOVERY_MODEL,
            )
        _validate_preflight_configuration(
            configuration,
            settings,
            expected_coding_model=expected_coding_model,
            expected_abstraction_model=expected_abstraction_model,
        )
        environments, frozen_cases = _preflight(
            project_root=project_root,
            baseline_root=baseline_root,
            execution_root=execution_root,
            configuration=configuration,
            cases=cases,
        )
        configuration_hash = _r2_configuration_hash(configuration, settings, environments)
        manifest["configuration_hash"] = configuration_hash
        manifest["coding_model"] = settings.openrouter_coding_model
        manifest["abstraction_model"] = settings.openrouter_abstraction_model
        manifest["prompt_version"] = configuration.model.prompt_version
        manifest["limits"] = configuration.config.limits.model_dump(mode="json")
        if resume_root is not None:
            existing_raw: object = json.loads(
                (artifact_root / "manifest.json").read_text(encoding="utf-8")
            )
            if not isinstance(existing_raw, dict):
                raise GateA1AcquisitionPreflightError(
                    f"{_revision} resume manifest is not an object"
                )
            existing = cast(dict[str, object], existing_raw)
            resume_validation_failed = True
            if _revision == "R5":
                manifest_validator(
                    existing,
                    configuration_hash=configuration_hash,
                    coding_model=settings.openrouter_coding_model,
                    abstraction_model=settings.openrouter_abstraction_model,
                )
            else:
                manifest_validator(existing, configuration_hash=configuration_hash)
            resume_validation_failed = False
            manifest = existing
        else:
            _write_json(artifact_root / "manifest.json", manifest)

        repository = Neo4jRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        repository.verify_connectivity()
        repository.ensure_recovery_pattern_vector_index()
        objective = FrozenSWEsmithObjective(frozen_cases, environments)
        pacing = ProviderRequestPacing()
        embedder: RecoveryPatternEmbedder | None = None
        task_records: list[dict[str, object]] = []
        task_plan = dict(
            _r2_task_plan(artifact_root, max_new_tasks=max_new_tasks)
        )
        new_tasks_started = 0
        boundary_hit = False
        for case in cases:
            task_id = case.task.id
            resume_action = task_plan.get(task_id)
            if resume_action is None:
                boundary_hit = True
                manifest["next_task_id"] = task_id
                break
            completed_marker = _r2_task_marker(artifact_root, task_id, "completed.json")
            started_marker = _r2_task_marker(artifact_root, task_id, "started.json")
            if resume_action == "skipped_completed":
                completed_raw: object = json.loads(
                    completed_marker.read_text(encoding="utf-8")
                )
                completed = (
                    cast(dict[str, object], completed_raw)
                    if isinstance(completed_raw, dict)
                    else {}
                )
                task_records.append(
                    {"task_id": task_id, "chronological_index": case.task.chronological_index,
                     "resume_action": resume_action, "status": completed.get("status")}
                )
                continue
            if resume_action == "skipped_started_no_rerun":
                task_records.append(
                    {"task_id": task_id, "chronological_index": case.task.chronological_index,
                     "resume_action": resume_action}
                )
                continue
            run_id = f"GS-E003-A1-{_revision}-{task_id}-{uuid.uuid4().hex}"
            _write_json(
                started_marker,
                {
                    "task_id": task_id,
                    "chronological_index": case.task.chronological_index,
                    "run_id": run_id,
                    "configuration_hash": configuration_hash,
                    "started_at": datetime.now(UTC),
                },
            )
            task_artifact, embedder, _ = _run_task(
                baseline_root=baseline_root,
                execution_root=execution_root,
                artifact_root=artifact_root,
                configuration=configuration,
                case=case,
                environment=environments[task_id],
                objective=objective,
                repository=repository,
                settings=settings,
                pacing=pacing,
                embedder=embedder,
                run_id=run_id,
                namespace=namespace,
                condition=condition,
                artifact_subdirectory=task_artifact_subdirectory,
            )
            task_records.append(task_artifact)
            _write_json(
                completed_marker,
                {
                    "task_id": task_id,
                    "chronological_index": case.task.chronological_index,
                    "run_id": run_id,
                    "configuration_hash": configuration_hash,
                    "status": task_artifact.get("status"),
                    "completed_at": datetime.now(UTC),
                },
            )
            new_tasks_started += 1
            manifest["new_tasks_started"] = new_tasks_started
            manifest["tasks"] = task_records
            _write_json(artifact_root / "manifest.json", manifest)
        manifest["tasks"] = task_records
        manifest["new_tasks_started"] = new_tasks_started
        manifest["completed_tasks"] = _completed_task_count(artifact_root)
        blocked = [
            str(record.get("task_id"))
            for record in task_records
            if record.get("resume_action") == "skipped_started_no_rerun"
            or (
                "status" in record
                and not str(record.get("status")).startswith("acquired")
            )
        ]
        boundary_hit = _reconcile_execution_boundary(
            artifact_root,
            manifest,
            boundary_hit=boundary_hit,
        )
        if boundary_hit:
            manifest["status"] = ready_to_resume_status
            manifest["errors"] = []
        elif len(task_records) == len(ACQUISITION_TASK_IDS) and not blocked:
            manifest["status"] = "READY_FOR_GATE_A1_RETRIEVAL_EVALUATION"
        else:
            manifest["errors"] = [{"blocked_tasks": blocked}]
    except Exception as error:
        if resume_validation_failed:
            raise
        manifest["errors"] = [{"type": type(error).__name__, "message": str(error)}]
    finally:
        if repository is not None:
            repository.close()
    _write_json(artifact_root / "manifest.json", manifest)
    return str(manifest["status"]), artifact_root


def run_gate_a1_acquisition_r3(
    project_root: Path,
    *,
    resume_root: Path | None = None,
    max_new_tasks: int | None = None,
) -> tuple[str, Path]:
    """Run or explicitly resume the fresh Gate A1 Acquisition R3 revision."""
    return run_gate_a1_acquisition_r2(
        project_root,
        resume_root=resume_root,
        max_new_tasks=max_new_tasks,
        _revision="R3",
    )


def run_gate_a1_acquisition_r4(
    project_root: Path,
    *,
    resume_root: Path | None = None,
    max_new_tasks: int | None = None,
) -> tuple[str, Path]:
    """Run or explicitly resume the budget-only Gate A1 Acquisition R4 revision."""
    return run_gate_a1_acquisition_r2(
        project_root,
        resume_root=resume_root,
        max_new_tasks=max_new_tasks,
        _revision="R4",
    )


def run_gate_a1_acquisition_r5(
    project_root: Path,
    *,
    resume_root: Path | None = None,
    max_new_tasks: int | None = None,
) -> tuple[str, Path]:
    """Run or explicitly resume the role-configured Gate A1 Acquisition R5 revision."""
    return run_gate_a1_acquisition_r2(
        project_root,
        resume_root=resume_root,
        max_new_tasks=max_new_tasks,
        _revision="R5",
    )


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
    except Exception as error:
        manifest["errors"] = [{"type": type(error).__name__, "message": str(error)}]
    _write_json(artifact_root / "manifest.json", manifest)
    return str(manifest["status"]), artifact_root


__all__ = [
    "run_gate_a1_acquisition",
    "run_gate_a1_acquisition_r2",
    "run_gate_a1_acquisition_r3",
    "run_gate_a1_acquisition_r4",
    "run_gate_a1_acquisition_r5",
]
