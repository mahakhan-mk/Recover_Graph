"""Gate A1 R13 applicability-key-corrected acquisition."""
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from graph_swarm.agent.stagnation import PreMutationStagnationGuard
from graph_swarm.graph.neo4j_repository import EntityNotFoundError
from graph_swarm.memory.recovery_abstraction import (
    OBJECTIVE_ANCHORED_RECOVERY_EVIDENCE_SOURCE,
    abstract_and_persist_recovery_pattern,
    build_recovery_evidence_package,
    deterministic_recovery_pattern_id,
)
from graph_swarm.memory.recovery_embeddings import (
    RecoveryPatternEmbedder,
    embed_and_persist_recovery_pattern,
)
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
    R13B_EXPECTED_ABSTRACTION_MODEL,
    R13B_EXPECTED_CODING_MODEL,
    _preflight,
    _r2_task_marker,
    _r8_configuration_hash,
    _settings_for_agent,
    _task_cases,
    _validate_preflight_configuration,
    _validate_r13_harness_configuration,
    _write_json,
    pre_mutation_stagnation_provenance,
    r13_runtime_timeout_resolutions,
    r13b_pre_mutation_stagnation_resolution,
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

R13B_RETRY_POLICY = "single_explicit_failed_task_retry_v1"
R13B_RETRY_REASON = "extended_agent_wall_clock_after_documented_timeout"
R13B_PATTERN_RETRY_POLICY = "single_explicit_pattern_retry_v1"
R13B_PATTERN_RETRY_REASON = "retry_recovery_abstraction_after_structured_output_failure"
R13B_MANIFEST_REFRESH_MARKER = (
    "READY_TO_REFRESH_R13B_MANIFEST_AND_PROCEED_TO_GS_T005"
)


def _r13b_retry_root(artifact_root: Path, task_id: str) -> Path:
    return artifact_root / "tasks" / task_id / "retries" / "attempt-001"


def _r13b_retry_eligibility(
    artifact_root: Path,
    task_id: str | None,
) -> dict[str, object]:
    """Validate and return the immutable initial attempt for one R13b retry."""
    if task_id is None:
        raise ValueError("--retry-r13b-task requires a task ID")
    if task_id not in ACQUISITION_TASK_IDS:
        raise ValueError(f"R13b retry task is not a frozen acquisition task: {task_id}")
    completed_path = _r2_task_marker(artifact_root, task_id, "completed.json")
    if not completed_path.is_file():
        raise ValueError(
            f"R13b retry requires a completed initial attempt for {task_id}"
        )
    try:
        initial_raw: object = json.loads(completed_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"R13b initial completed attempt is unreadable for {task_id}"
        ) from error
    if not isinstance(initial_raw, dict):
        raise ValueError(f"R13b initial completed attempt is invalid for {task_id}")
    initial = cast(dict[str, object], initial_raw)
    if initial.get("acquisition_success") is not False:
        raise ValueError(
            f"R13b retry requires an unsuccessful completed attempt for {task_id}"
        )
    retry_root = _r13b_retry_root(artifact_root, task_id)
    if retry_root.exists() or retry_root.parent.exists() and any(retry_root.parent.iterdir()):
        raise ValueError(f"R13b controlled retry already consumed for {task_id}")
    return initial


def _r13b_attempt_summary(
    record: Mapping[str, object],
    *,
    attempt: int,
    kind: str,
) -> dict[str, object]:
    return {
        "attempt": attempt,
        "kind": kind,
        "run_id": record.get("run_id"),
        "acquisition_success": bool(record.get("acquisition_success")),
    }


def _r13b_selected_task_record(
    artifact_root: Path,
    task_id: str,
    manifest: Mapping[str, object],
) -> dict[str, object] | None:
    pattern_retry = manifest.get("pattern_retry")
    if isinstance(pattern_retry, dict) and pattern_retry.get("task_id") == task_id:
        pattern_retry_root = (
            artifact_root / "tasks" / task_id / "pattern-retries" / "attempt-001"
        )
        completed_path = pattern_retry_root / "completed.json"
        if completed_path.is_file():
            raw: object = json.loads(completed_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                selected = raw.get("selected_task_record")
                if isinstance(selected, dict):
                    return cast(dict[str, object], selected)
    retry_root = _r13b_retry_root(artifact_root, task_id)
    completed_path = retry_root / "completed.json"
    if completed_path.is_file():
        raw = json.loads(completed_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            retry = cast(dict[str, object], raw)
            if (
                retry.get("task_id") == task_id
                and retry.get("attempt") == 2
                and retry.get("kind") == "controlled_retry"
            ):
                return retry
    completed_path = _r2_task_marker(artifact_root, task_id, "completed.json")
    if completed_path.is_file():
        raw = json.loads(completed_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return cast(dict[str, object], raw)
    return None


def _r13b_effective_task_records(
    artifact_root: Path,
    manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return one selected completed record for each frozen logical task ID."""
    return [
        selected
        for task_id in ACQUISITION_TASK_IDS
        if (selected := _r13b_selected_task_record(artifact_root, task_id, manifest))
        is not None
    ]


def refresh_r13b_manifest(artifact_root: Path) -> tuple[str, Path]:
    """Refresh R13b bookkeeping from stored artifacts only.

    This path deliberately does not load experiment configuration or construct
    agent, provider, abstraction, embedding, workspace, or Neo4j components.
    """
    artifact_root = artifact_root.expanduser().resolve()
    if not artifact_root.name.startswith("acquisition-r13b-"):
        raise ValueError("R13b manifest refresh requires an acquisition-r13b-* root")
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("R13b manifest refresh root is missing manifest.json")
    raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("R13b manifest is not an object")
    manifest = cast(dict[str, object], raw_manifest)
    if manifest.get("run_revision") != "R13b":
        raise ValueError("R13b manifest refresh requires a R13b root")
    if manifest.get("config_version") != "gate-a1-r13b-v2":
        raise ValueError("R13b manifest refresh requires config_version=gate-a1-r13b-v2")

    task_records = _r13b_effective_task_records(artifact_root, manifest)
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
    manifest["tasks"] = task_records
    manifest["completed_tasks"] = len(completed_task_ids)
    manifest["completed_task_ids"] = list(completed_task_ids)
    manifest["eligible_acquisition_tasks"] = eligible_acquisition_tasks
    manifest["acquisition_corpus_ready"] = corpus_ready
    manifest["status"] = status.replace("_R8", "_R13B")
    _write_json(manifest_path, manifest)
    return R13B_MANIFEST_REFRESH_MARKER, artifact_root


def _r13b_pattern_retry_root(artifact_root: Path, task_id: str) -> Path:
    return artifact_root / "tasks" / task_id / "pattern-retries" / "attempt-001"


def _r13b_pattern_retry_eligibility(
    artifact_root: Path,
    task_id: str | None,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate a downstream-only retry from one selected R13b task artifact."""
    if task_id is None:
        raise ValueError("--retry-r13b-pattern requires a task ID")
    if task_id not in ACQUISITION_TASK_IDS:
        raise ValueError(f"R13b pattern retry task is not frozen: {task_id}")
    pattern_retry_root = _r13b_pattern_retry_root(artifact_root, task_id)
    if pattern_retry_root.exists() or (
        pattern_retry_root.parent.exists() and any(pattern_retry_root.parent.iterdir())
    ):
        raise ValueError(f"R13b pattern retry already consumed for {task_id}")
    if isinstance(manifest.get("pattern_retry"), dict):
        prior = cast(dict[str, object], manifest["pattern_retry"])
        if prior.get("task_id") == task_id:
            raise ValueError(f"R13b pattern retry already recorded for {task_id}")
    selected = _r13b_selected_task_record(artifact_root, task_id, manifest)
    if selected is None:
        raise ValueError(
            f"R13b pattern retry requires a completed selected attempt for {task_id}"
        )
    if selected.get("task_success") is not True:
        raise ValueError("R13b pattern retry requires a successful task objective")
    if selected.get("complete_trusted_lineage") is not True:
        raise ValueError("R13b pattern retry requires complete trusted lineage")
    counts = selected.get("counts")
    if not isinstance(counts, dict) or not isinstance(counts.get("recoveries"), int):
        raise ValueError("R13b pattern retry requires persisted recovery evidence")
    if counts["recoveries"] < 1:
        raise ValueError("R13b pattern retry requires at least one recovery")
    if selected.get("recovery_evidence_source") != OBJECTIVE_ANCHORED_RECOVERY_EVIDENCE_SOURCE:
        raise ValueError("R13b pattern retry requires objective-anchored recovery evidence")
    if (
        selected.get("pattern_created") is True
        and selected.get("pattern_embedded") is True
        and selected.get("pattern_persisted") is True
    ):
        raise ValueError(f"R13b pattern already fully acquired for {task_id}")
    lineage = selected.get("recovery_lineage")
    if not isinstance(lineage, dict):
        raise ValueError("R13b pattern retry requires canonical recovery lineage IDs")
    for field in ("failure_id", "resolution_id", "outcome_id"):
        if not isinstance(lineage.get(field), str) or not lineage[field].strip():
            raise ValueError(f"R13b recovery lineage is missing {field}")
    trigger = selected.get("objective_success_trigger_action_id")
    if not isinstance(trigger, str) or not trigger.strip():
        raise ValueError("R13b pattern retry requires a trusted recovery action")
    return selected


def _r13b_pattern_retry_task_record(
    original: Mapping[str, object],
    retry: Mapping[str, object],
) -> dict[str, object]:
    selected = dict(original)
    for field in (
        "pattern_created",
        "pattern_embedded",
        "pattern_persisted",
        "persistence_error",
    ):
        selected[field] = retry.get(field)
    selected["acquisition_success"] = bool(
        retry.get("acquisition_success_after_pattern_retry")
    )
    selected["acquisition_reason"] = (
        "complete_trusted_recovery_lineage_and_pattern"
        if selected["acquisition_success"]
        else "recovery_pattern_not_created"
        if not selected["pattern_created"]
        else "recovery_pattern_not_embedded"
        if not selected["pattern_embedded"]
        else "recovery_pattern_not_persisted"
    )
    selected["pattern_retry"] = dict(retry)
    return selected


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
    retry_task_id: str | None = None,
) -> tuple[str, Path]:
    """Execute R13 or a thin R13-compatible revision."""
    from experiments.sprint3 import FrozenSWEsmithObjective, _configured_runtime
    from graph_swarm.agent.pacing import ProviderRequestPacing
    from graph_swarm.graph.neo4j_repository import Neo4jRepository
    from graph_swarm.research.runner import load_experiment_configuration

    project_root = project_root.expanduser().resolve()
    if max_new_tasks is not None and max_new_tasks <= 0:
        raise ValueError("max_new_tasks must be positive")
    if retry_task_id is not None:
        if _revision != "R13b":
            raise ValueError("--retry-r13b-task is valid only for R13b")
        if resume_root is None:
            raise ValueError("--retry-r13b-task requires --resume-acquisition-r13b")
        if max_new_tasks is not None:
            raise ValueError("--retry-r13b-task cannot be combined with --max-new-tasks")

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
    pre_mutation_config = (
        r13b_pre_mutation_stagnation_resolution() if _revision == "R13b" else None
    )
    initial_attempt: dict[str, object] | None = None

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
        if retry_task_id is not None:
            initial_attempt = _r13b_retry_eligibility(artifact_root, retry_task_id)
        else:
            initial_attempt = None

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
        if pre_mutation_config is not None:
            manifest.update(pre_mutation_stagnation_provenance(pre_mutation_config))
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
        if pre_mutation_config is not None:
            manifest.update(pre_mutation_stagnation_provenance(pre_mutation_config))
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
    if retry_task_id is not None:
        assert initial_attempt is not None
        retry_case = next(case for case in cases if case.task.id == retry_task_id)
        retry_root = _r13b_retry_root(artifact_root, retry_task_id)
        retry_root.mkdir(parents=True, exist_ok=False)
        run_id = f"GS-E003-A1-{_revision}-{retry_task_id}-{uuid.uuid4().hex}"
        retry_started = retry_root / "started.json"
        _write_json(
            retry_started,
            {
                "task_id": retry_task_id,
                "run_id": run_id,
                "attempt": 2,
                "kind": "controlled_retry",
                "configuration_hash": configuration_hash,
            },
        )
        task_artifact, embedder = _run_r8_task(
            project_root=project_root,
            artifact_root=artifact_root,
            configuration=configuration,
            case=retry_case,
            environment=environments[retry_task_id],
            objective=objective,
            settings=settings,
            pacing=ProviderRequestPacing(),
            embedder=embedder,
            run_id=run_id,
            memory=memory,
            revision=_revision,
            condition=f"{_condition}-retry-001",
            controller_type=_controller_type,
            objective_anchored_acquisition=True,
            system_prompt=_system_prompt,
            agent_timeout_seconds=timeout_resolutions["agent"].effective_seconds,
            objective_timeout_seconds=timeout_resolutions["objective"].effective_seconds,
            runtime_timeout_overrides=runtime_timeout_provenance(timeout_resolutions),
            attempt_number=2,
            attempt_kind="controlled_retry",
            pre_mutation_guard=(
                None
                if pre_mutation_config is None
                else PreMutationStagnationGuard(pre_mutation_config)
            ),
        )
        retry_completed = dict(task_artifact)
        retry_completed.update({"attempt": 2, "kind": "controlled_retry"})
        _write_json(retry_root / "completed.json", retry_completed)
        manifest["retry"] = {
            "policy": R13B_RETRY_POLICY,
            "reason": R13B_RETRY_REASON,
            "task_id": retry_task_id,
            "selected_attempt": 2,
            "attempts": [
                _r13b_attempt_summary(
                    initial_attempt,
                    attempt=1,
                    kind="initial",
                ),
                _r13b_attempt_summary(retry_completed, attempt=2, kind="controlled_retry"),
            ],
        }
        task_records = _r13b_effective_task_records(artifact_root, manifest)
    else:
        for case in cases:
            task_id = case.task.id
            started_marker = _r2_task_marker(artifact_root, task_id, "started.json")
            selected = _r13b_selected_task_record(artifact_root, task_id, manifest)
            if selected is not None:
                task_records.append(selected)
                continue
            if started_marker.is_file():
                task_records.append(
                    {"task_id": task_id, "resume_action": "skipped_started_no_rerun"}
                )
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
                pre_mutation_guard=(
                    None
                    if pre_mutation_config is None
                    else PreMutationStagnationGuard(pre_mutation_config)
                ),
            )
            task_records.append(task_artifact)
            _write_json(
                _r2_task_marker(artifact_root, task_id, "completed.json"),
                task_artifact,
            )
            started_new += 1
            manifest["tasks"] = task_records
            _write_json(manifest_path, manifest)

    task_records = _r13b_effective_task_records(artifact_root, manifest)
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


def run_gate_a1_pattern_retry_r13b(
    project_root: Path,
    *,
    resume_root: Path,
    task_id: str,
    _configuration: Any | None = None,
) -> tuple[str, Path]:
    """Retry only R13b recovery-pattern downstream processing for one task."""
    from experiments.sprint3 import _configured_runtime
    from graph_swarm.research.runner import load_experiment_configuration

    project_root = project_root.expanduser().resolve()
    artifact_root = resume_root.expanduser().resolve()
    if not artifact_root.name.startswith("acquisition-r13b-"):
        raise ValueError("R13b pattern retry requires an acquisition-r13b-* root")
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("R13b pattern retry root is missing manifest.json")
    raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("R13b pattern retry manifest is not an object")
    manifest = cast(dict[str, object], raw_manifest)
    if manifest.get("run_revision") != "R13b":
        raise ValueError("R13b pattern retry requires a R13b resume root")
    if manifest.get("config_version") != "gate-a1-r13b-v2":
        raise ValueError("R13b pattern retry requires config_version=gate-a1-r13b-v2")

    if _configuration is None:
        configuration = _configured_runtime(
            load_experiment_configuration(
                project_root / "configs/experiments/gate_a1_acquisition_r13b_v2.yaml",
                project_root=project_root,
            ),
            project_root / "benchmark/workspaces",
            project_root / "research/evidence/workspaces",
        )
    else:
        configuration = _configuration
    if (
        configuration.config.run_revision != "R13b"
        or configuration.config.config_version != "gate-a1-r13b-v2"
        or configuration.model.prompt_version != "v2-runtime-guidance"
    ):
        raise ValueError("R13b pattern retry requires the frozen R13b configuration")
    selected = _r13b_pattern_retry_eligibility(artifact_root, task_id, manifest)

    from graph_swarm.graph.neo4j_repository import Neo4jRepository

    settings = _settings_for_agent(
        coding_model=R13B_EXPECTED_CODING_MODEL,
        abstraction_model=R13B_EXPECTED_ABSTRACTION_MODEL,
    )
    if settings.openrouter_abstraction_model != R13B_EXPECTED_ABSTRACTION_MODEL:
        raise ValueError("R13b abstraction model differs from the frozen model")
    retry_root = _r13b_pattern_retry_root(artifact_root, task_id)
    retry_root.mkdir(parents=True, exist_ok=False)
    original_run_id = selected.get("run_id")
    if not isinstance(original_run_id, str) or not original_run_id.strip():
        raise ValueError("R13b pattern retry requires the original task run ID")
    _write_json(
        retry_root / "started.json",
        {
            "task_id": task_id,
            "original_run_id": original_run_id,
            "pattern_retry_attempt": 1,
            "pattern_retry_policy": R13B_PATTERN_RETRY_POLICY,
            "pattern_retry_reason": R13B_PATTERN_RETRY_REASON,
            "coding_agent_rerun": False,
        },
    )

    def repository_factory() -> Neo4jRepository:
        return Neo4jRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )

    memory = ShortLivedNeo4jRepository(repository_factory)
    memory.verify_connectivity()
    lineage_ids = cast(dict[str, object], selected["recovery_lineage"])
    failure_id = cast(str, lineage_ids["failure_id"])
    resolution_id = cast(str, lineage_ids["resolution_id"])
    outcome_id = cast(str, lineage_ids["outcome_id"])
    objective_trigger = cast(str, selected["objective_success_trigger_action_id"])
    lineage = memory.get_recovery_evidence(failure_id)
    if (
        lineage.failure.id != failure_id
        or lineage.resolution.id != resolution_id
        or lineage.outcome.id != outcome_id
        or lineage.recovery_action.planned_action.id != objective_trigger
    ):
        raise ValueError("R13b canonical recovery evidence does not match the task artifact")
    lineage = lineage.model_copy(
        update={
            "recovery_evidence_source": OBJECTIVE_ANCHORED_RECOVERY_EVIDENCE_SOURCE,
            "trusted_recovery_action_id": objective_trigger,
        }
    )
    evidence = build_recovery_evidence_package(lineage)
    pattern_id = deterministic_recovery_pattern_id(evidence)
    pattern_created = False
    pattern_persisted = False
    pattern_embedded = False
    persistence_error: str | None = None
    persistence_error_message: str | None = None
    pattern: Any = None
    try:
        try:
            existing = memory.get_recovery_pattern(pattern_id)
        except EntityNotFoundError:
            existing = None
        if existing is not None:
            pattern = existing.pattern
            pattern_created = True
            pattern_persisted = True
            pattern_embedded = pattern.embedding is not None
        else:
            pattern = abstract_and_persist_recovery_pattern(
                lineage,
                cast(Any, memory),
                settings,
            )
            pattern_created = True
            pattern_persisted = True
        if pattern is not None and not pattern_embedded:
            embedded = embed_and_persist_recovery_pattern(
                pattern,
                cast(Any, memory),
                RecoveryPatternEmbedder(),
            )
            pattern_embedded = embedded.embedding is not None
    except Exception as error:
        persistence_error = type(error).__name__
        persistence_error_message = str(error)[:500]

    acquisition_success = bool(pattern_created and pattern_embedded and pattern_persisted)
    retry_record: dict[str, object] = {
        "task_id": task_id,
        "original_run_id": original_run_id,
        "pattern_retry_attempt": 1,
        "pattern_retry_policy": R13B_PATTERN_RETRY_POLICY,
        "pattern_retry_reason": R13B_PATTERN_RETRY_REASON,
        "coding_agent_rerun": False,
        "workspace_recreated": False,
        "objective_evaluator_rerun": False,
        "recovery_evidence_reused": True,
        "recovery_evidence_source": OBJECTIVE_ANCHORED_RECOVERY_EVIDENCE_SOURCE,
        "source_failure_id": failure_id,
        "source_resolution_id": resolution_id,
        "source_outcome_id": outcome_id,
        "trusted_recovery_action_id": objective_trigger,
        "objective_success_trigger_action_id": objective_trigger,
        "source_tool": lineage.recovery_action.planned_action.tool,
        "source_operation": lineage.recovery_action.planned_action.operation,
        "applicability_tool": lineage.recovery_action.planned_action.tool,
        "applicability_operation": lineage.recovery_action.planned_action.operation,
        "abstraction_model": settings.openrouter_abstraction_model,
        "pattern_id": pattern_id,
        "pattern_created": pattern_created,
        "pattern_embedded": pattern_embedded,
        "pattern_persisted": pattern_persisted,
        "persistence_error": persistence_error,
        "persistence_error_message": persistence_error_message,
        "acquisition_success_after_pattern_retry": acquisition_success,
        "acquisition_reason": (
            "complete_trusted_recovery_lineage_and_pattern"
            if acquisition_success
            else "recovery_pattern_not_created"
            if not pattern_created
            else "recovery_pattern_not_embedded"
            if not pattern_embedded
            else "recovery_pattern_not_persisted"
        ),
    }
    selected_task_record = _r13b_pattern_retry_task_record(selected, retry_record)
    retry_record["selected_task_record"] = selected_task_record
    _write_json(retry_root / "completed.json", retry_record)
    manifest["pattern_retry"] = {
        "task_id": task_id,
        "policy": R13B_PATTERN_RETRY_POLICY,
        "reason": R13B_PATTERN_RETRY_REASON,
        "original_run_id": original_run_id,
        "attempts": [
            {
                "attempt": 1,
                "kind": "pattern_retry",
                "pattern_created": pattern_created,
                "pattern_embedded": pattern_embedded,
                "pattern_persisted": pattern_persisted,
                "acquisition_success": acquisition_success,
            }
        ],
        "selected_attempt": 1,
    }
    task_records = _r13b_effective_task_records(artifact_root, manifest)
    completed_task_ids = tuple(
        candidate
        for candidate in ACQUISITION_TASK_IDS
        if _r2_task_marker(artifact_root, candidate, "completed.json").is_file()
    )
    status, corpus_ready, eligible_acquisition_tasks = r8_acquisition_readiness(
        task_records,
        completed_task_ids,
        ACQUISITION_TASK_IDS,
    )
    status = status.replace("_R8", "_R13B")
    manifest["tasks"] = task_records
    manifest["completed_tasks"] = len(completed_task_ids)
    manifest["completed_task_ids"] = list(completed_task_ids)
    manifest["eligible_acquisition_tasks"] = eligible_acquisition_tasks
    manifest["acquisition_corpus_ready"] = corpus_ready
    manifest["status"] = status
    _write_json(manifest_path, manifest)
    memory.close()
    return str(status), artifact_root


__all__ = [
    "R13ObjectiveController",
    "run_gate_a1_acquisition_r13",
    "run_gate_a1_pattern_retry_r13b",
    "refresh_r13b_manifest",
    "R13B_MANIFEST_REFRESH_MARKER",
]
