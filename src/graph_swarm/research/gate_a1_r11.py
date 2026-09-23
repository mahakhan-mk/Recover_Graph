"""Gate A1 R11 runner restoring the frozen Track A coding model."""
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from graph_swarm.graph.neo4j_repository import Neo4jRepository
from graph_swarm.research.gate_a1_acquisition import (
    ACQUISITION_TASK_IDS,
    R9_AGENT_TIMEOUT_SECONDS,
    R9_OBJECTIVE_TIMEOUT_SECONDS,
    R9_RECOVERY_EVENT_SEMANTICS,
    R9_REPOSITORY_MUTATION_EVIDENCE_POLICY,
    R10_COMMAND_ARGV_POLICY,
    R11_CONFIG,
    R11_EXPECTED_ABSTRACTION_MODEL,
    R11_EXPECTED_CODING_MODEL,
    R11_NAMESPACE,
    R11_REVISION_REASON,
    _preflight,
    _r2_task_marker,
    _r8_configuration_hash,
    _settings_for_agent,
    _task_cases,
    _validate_preflight_configuration,
    _validate_r11_harness_configuration,
    _write_json,
)
from graph_swarm.research.gate_a1_r8 import (
    R8_LINE_ENDING_POLICY,
    R8_PERSISTENCE_RETRY_POLICY,
    R8_PERSISTENCE_SESSION_POLICY,
    R8_STOPPING_POLICY,
    ShortLivedNeo4jRepository,
    _run_r8_task,
    r8_acquisition_readiness,
)
from graph_swarm.research.gate_a1_r10 import R10ObjectiveController


class R11ObjectiveController(R10ObjectiveController):
    """R11 preserves R10 runtime and R9 trusted recovery semantics unchanged."""


def run_gate_a1_acquisition_r11(
    project_root: Path,
    *,
    resume_root: Path | None = None,
    max_new_tasks: int | None = None,
) -> tuple[str, Path]:
    """Execute R11 only when explicitly requested."""
    from experiments.sprint3 import FrozenSWEsmithObjective, _configured_runtime
    from graph_swarm.agent.pacing import ProviderRequestPacing
    from graph_swarm.research.runner import load_experiment_configuration

    project_root = project_root.expanduser().resolve()
    if max_new_tasks is not None and max_new_tasks <= 0:
        raise ValueError("max_new_tasks must be positive")
    run_prefix = "acquisition-r11"
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
            raise ValueError("R11 resume path must use an acquisition-r11-* namespace")
        manifest_path = artifact_root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("R11 resume root is missing manifest.json")
        existing: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("run_revision") != "R11":
            raise ValueError("R11 rejects resume roots from R10, R9, R8, or older revisions")

    configuration = _configured_runtime(
        load_experiment_configuration(
            project_root / R11_CONFIG,
            project_root=project_root,
        ),
        project_root / "benchmark/workspaces",
        project_root / "research/evidence/workspaces",
    )
    _validate_r11_harness_configuration(configuration)
    cases = _task_cases(configuration, ACQUISITION_TASK_IDS)
    settings = _settings_for_agent(
        coding_model=R11_EXPECTED_CODING_MODEL,
        abstraction_model=R11_EXPECTED_ABSTRACTION_MODEL,
    )
    _validate_preflight_configuration(
        configuration,
        settings,
        expected_coding_model=R11_EXPECTED_CODING_MODEL,
        expected_abstraction_model=R11_EXPECTED_ABSTRACTION_MODEL,
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
        coverage_policy_selection_version=(
            configuration.config.objective_coverage_policy_selection_version
        ),
        objective_timeout_seconds=cast(
            float, configuration.config.objective_timeout_seconds
        ),
    )
    configuration_hash = _r8_configuration_hash(configuration, settings, environments)
    manifest: dict[str, object] = {
        "gate": "GS-E003 / Gate A1",
        "phase": "acquisition",
        "run_revision": "R11",
        "namespace": R11_NAMESPACE,
        "task_ids": list(ACQUISITION_TASK_IDS),
        "status": "BLOCKED_GATE_A1_ACQUISITION_R11",
        "configuration_hash": configuration_hash,
        "coding_model": settings.openrouter_coding_model,
        "abstraction_model": settings.openrouter_abstraction_model,
        "prompt_version": configuration.model.prompt_version,
        "limits": configuration.config.limits.model_dump(mode="json"),
        "agent_timeout_seconds": R9_AGENT_TIMEOUT_SECONDS,
        "objective_timeout_seconds": R9_OBJECTIVE_TIMEOUT_SECONDS,
        "stopping_policy": R8_STOPPING_POLICY,
        "objective_mutation_check_policy": "repository_state_fingerprint_v1",
        "workspace_line_ending_policy": R8_LINE_ENDING_POLICY,
        "persistence_session_policy": R8_PERSISTENCE_SESSION_POLICY,
        "persistence_retry_policy": R8_PERSISTENCE_RETRY_POLICY,
        "recovery_event_semantics": R9_RECOVERY_EVENT_SEMANTICS,
        "repository_mutation_evidence_policy": R9_REPOSITORY_MUTATION_EVIDENCE_POLICY,
        "command_argv_policy": R10_COMMAND_ARGV_POLICY,
        "revision_reason": R11_REVISION_REASON,
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
        if not isinstance(existing, dict) or existing.get("run_revision") != "R11":
            raise ValueError("R11 rejects resume roots from R10, R9, R8, or older revisions")
        if existing.get("configuration_hash") != configuration_hash:
            raise ValueError("R11 resume configuration hash differs")
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
            task_records.append(
                {"task_id": task_id, "resume_action": "skipped_started_no_rerun"}
            )
            continue
        if max_new_tasks is not None and started_new >= max_new_tasks:
            break
        run_id = f"GS-E003-A1-R11-{task_id}-{uuid.uuid4().hex}"
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
            revision="R11",
            condition="acquisition-r11",
            controller_type=R11ObjectiveController,
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
    status = status.replace("_R8", "_R11")
    manifest["completed_tasks"] = len(completed_task_ids)
    manifest["completed_task_ids"] = list(completed_task_ids)
    manifest["eligible_acquisition_tasks"] = eligible_acquisition_tasks
    manifest["acquisition_corpus_ready"] = corpus_ready
    manifest["status"] = status
    if status == "BLOCKED_GATE_A1_ACQUISITION_R11_NON_EVALUABLE":
        manifest["acquisition_reason"] = "zero_eligible_recovery_patterns"
    _write_json(manifest_path, manifest)
    return str(status), artifact_root


__all__ = ["R11ObjectiveController", "run_gate_a1_acquisition_r11"]
