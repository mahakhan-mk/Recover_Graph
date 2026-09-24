"""Gate A1 R13 applicability-key-corrected acquisition."""
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from graph_swarm.research.gate_a1_acquisition import (
    ACQUISITION_TASK_IDS,
    R9_REPOSITORY_MUTATION_EVIDENCE_POLICY,
    R10_COMMAND_ARGV_POLICY,
    R13_CONFIG,
    R13_EXPECTED_ABSTRACTION_MODEL,
    R13_EXPECTED_CODING_MODEL,
    R13_NAMESPACE,
    R13_OBJECTIVE_MUTATION_CHECK_POLICY,
    R13_RECOVERY_EVENT_SEMANTICS,
    R13_REVISION_REASON,
    R13_STOPPING_POLICY,
    _preflight,
    _r2_task_marker,
    _r8_configuration_hash,
    _settings_for_agent,
    _task_cases,
    _validate_preflight_configuration,
    _validate_r13_harness_configuration,
    _write_json,
    r13_runtime_timeout_resolutions,
    runtime_timeout_provenance,
)
from graph_swarm.research.gate_a1_r8 import (
    R8_LINE_ENDING_POLICY,
    R8_PERSISTENCE_RETRY_POLICY,
    R8_PERSISTENCE_SESSION_POLICY,
    ShortLivedNeo4jRepository,
    _run_r8_task,
    r8_acquisition_readiness,
)
from graph_swarm.research.gate_a1_r12 import R12ObjectiveController


class R13ObjectiveController(R12ObjectiveController):
    """R12 objective lifecycle with R13 pattern applicability semantics."""


def run_gate_a1_acquisition_r13(
    project_root: Path,
    *,
    resume_root: Path | None = None,
    max_new_tasks: int | None = None,
    _configuration: Any | None = None,
    _revision: str = "R13",
    _run_prefix: str = "acquisition-r13",
    _namespace: str = R13_NAMESPACE,
    _condition: str = "acquisition-r13",
    _controller_type: type[Any] = R13ObjectiveController,
    _system_prompt: str | None = None,
    _coding_model: str = R13_EXPECTED_CODING_MODEL,
    _abstraction_model: str = R13_EXPECTED_ABSTRACTION_MODEL,
) -> tuple[str, Path]:
    """Execute R13 or a thin R13-compatible revision."""
    from experiments.sprint3 import FrozenSWEsmithObjective, _configured_runtime
    from graph_swarm.agent.pacing import ProviderRequestPacing
    from graph_swarm.graph.neo4j_repository import Neo4jRepository
    from graph_swarm.research.runner import load_experiment_configuration

    project_root = project_root.expanduser().resolve()
    if max_new_tasks is not None and max_new_tasks <= 0:
        raise ValueError("max_new_tasks must be positive")

    if _configuration is None:
        configuration = _configured_runtime(
            load_experiment_configuration(
                project_root / R13_CONFIG,
                project_root=project_root,
            ),
            project_root / "benchmark/workspaces",
            project_root / "research/evidence/workspaces",
        )
        _validate_r13_harness_configuration(configuration)
    else:
        configuration = _configuration
    timeout_resolutions = r13_runtime_timeout_resolutions(configuration)

    if resume_root is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        artifact_root = (
            project_root
            / "research/evidence/results/GS-E003/gate_a1"
            / f"{_run_prefix}-{stamp}"
        )
        artifact_root.mkdir(parents=True, exist_ok=False)
    else:
        artifact_root = resume_root.expanduser().resolve()
        if not artifact_root.name.startswith(f"{_run_prefix}-"):
            raise ValueError(
                f"{_revision} resume path must use an {_run_prefix}-* namespace"
            )
        manifest_path = artifact_root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("R13 resume root is missing manifest.json")
        existing: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("run_revision") != _revision:
            raise ValueError(
                f"{_revision} rejects resume roots from older revisions"
            )

    cases = _task_cases(configuration, ACQUISITION_TASK_IDS)
    settings = _settings_for_agent(
        coding_model=_coding_model,
        abstraction_model=_abstraction_model,
    )
    _validate_preflight_configuration(
        configuration,
        settings,
        expected_coding_model=_coding_model,
        expected_abstraction_model=_abstraction_model,
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
        objective_timeout_seconds=timeout_resolutions["objective"].effective_seconds,
    )
    configuration_hash = _r8_configuration_hash(configuration, settings, environments)
    manifest: dict[str, object] = {
        "gate": "GS-E003 / Gate A1",
        "phase": "acquisition",
        "run_revision": _revision,
        "namespace": _namespace,
        "task_ids": list(ACQUISITION_TASK_IDS),
        "status": f"BLOCKED_GATE_A1_ACQUISITION_{_revision.upper()}",
        "configuration_hash": configuration_hash,
        "coding_model": settings.openrouter_coding_model,
        "abstraction_model": settings.openrouter_abstraction_model,
        "prompt_version": configuration.model.prompt_version,
        "limits": configuration.config.limits.model_dump(mode="json"),
        "agent_timeout_seconds": timeout_resolutions["agent"].effective_seconds,
        "objective_timeout_seconds": timeout_resolutions["objective"].effective_seconds,
        "runtime_timeout_overrides": runtime_timeout_provenance(timeout_resolutions),
        "stopping_policy": configuration.config.stopping_policy or R13_STOPPING_POLICY,
        "objective_mutation_check_policy": (
            configuration.config.objective_mutation_check_policy
            or R13_OBJECTIVE_MUTATION_CHECK_POLICY
        ),
        "workspace_line_ending_policy": (
            configuration.config.workspace_line_ending_policy or R8_LINE_ENDING_POLICY
        ),
        "persistence_session_policy": (
            configuration.config.persistence_session_policy or R8_PERSISTENCE_SESSION_POLICY
        ),
        "persistence_retry_policy": (
            configuration.config.persistence_retry_policy or R8_PERSISTENCE_RETRY_POLICY
        ),
        "recovery_event_semantics": (
            configuration.config.recovery_event_semantics or R13_RECOVERY_EVENT_SEMANTICS
        ),
        "repository_mutation_evidence_policy": (
            configuration.config.repository_mutation_evidence_policy
            or R9_REPOSITORY_MUTATION_EVIDENCE_POLICY
        ),
        "command_argv_policy": (
            configuration.config.command_argv_policy or R10_COMMAND_ARGV_POLICY
        ),
        "revision_reason": configuration.config.revision_reason or R13_REVISION_REASON,
        "model_visible_tool_output_chars": configuration.config.model_visible_tool_output_chars,
        "objective_coverage_policy": configuration.config.objective_coverage_policy,
        "objective_coverage_policy_selection_version": (
            configuration.config.objective_coverage_policy_selection_version
        ),
        "tasks": [],
        "errors": [],
    }
    if _revision == "R13b":
        manifest.update(
            {
                "config_version": configuration.config.config_version,
                "revision_reason": configuration.config.revision_reason,
            }
        )
    manifest_path = artifact_root / "manifest.json"
    if resume_root is not None:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("run_revision") != _revision:
            raise ValueError(
                f"{_revision} rejects resume roots from older revisions"
            )
        if existing.get("configuration_hash") != configuration_hash:
            raise ValueError("R13 resume configuration hash differs")
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
        run_id = f"GS-E003-A1-{_revision}-{task_id}-{uuid.uuid4().hex}"
        started_marker.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            started_marker,
            {"task_id": task_id, "run_id": run_id, "configuration_hash": configuration_hash},
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
            revision=_revision,
            condition=_condition,
            controller_type=_controller_type,
            objective_anchored_acquisition=True,
            system_prompt=_system_prompt,
            agent_timeout_seconds=timeout_resolutions["agent"].effective_seconds,
            objective_timeout_seconds=timeout_resolutions["objective"].effective_seconds,
            runtime_timeout_overrides=runtime_timeout_provenance(timeout_resolutions),
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
    status = status.replace("_R8", f"_{_revision.upper()}")
    manifest["completed_tasks"] = len(completed_task_ids)
    manifest["completed_task_ids"] = list(completed_task_ids)
    manifest["eligible_acquisition_tasks"] = eligible_acquisition_tasks
    manifest["acquisition_corpus_ready"] = corpus_ready
    manifest["status"] = status
    if status == f"BLOCKED_GATE_A1_ACQUISITION_{_revision.upper()}_NON_EVALUABLE":
        manifest["acquisition_reason"] = "zero_eligible_recovery_patterns"
    _write_json(manifest_path, manifest)
    return str(status), artifact_root


__all__ = ["R13ObjectiveController", "run_gate_a1_acquisition_r13"]
