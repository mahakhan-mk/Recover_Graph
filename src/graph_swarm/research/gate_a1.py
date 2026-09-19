"""Development-only Gate A1 readiness and environment preparation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from neo4j import Driver, GraphDatabase
from pydantic import BaseModel, ConfigDict, Field

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.tasks import Task
from graph_swarm.memory.recovery_abstraction import (
    FROZEN_RECOVERY_MODEL,
    RECOVERY_ABSTRACTION_PROMPT_VERSION,
)
from graph_swarm.memory.recovery_embeddings import (
    RECOVERY_PATTERN_EMBEDDING_DIMENSION,
    RECOVERY_PATTERN_EMBEDDING_MODEL,
)
from graph_swarm.retrieval.service import (
    RECOVERY_RETRIEVAL_QUERY_VERSION,
    RecoveryPatternRetrievalService,
)
from graph_swarm.settings import get_settings

GATE_A1 = "GS-E003 / Gate A1"
DEVELOPMENT_SUBSET_PATH = Path("configs/research/gate_a1_development_subset.json")
BASELINE_MANIFEST_PATH = Path("configs/research/gate_a1_repositories.json")
CANONICAL_ACQUISITION_TASK_IDS = tuple(f"GS-T{i:03d}" for i in range(1, 6))
CANONICAL_TRANSFER_TASK_IDS = tuple(f"GS-T{i:03d}" for i in range(6, 16))
VECTOR_INDEX = "recovery_pattern_embedding_idx"
ACQUISITION_CONTRACT = Path("configs/research/gate_a1_docker_environments.json")
ACQUISITION_POLICY = Path("configs/research/gate_a1_benchmark_environments.toml")
ACQUISITION_ENVIRONMENT_ROOT = Path("research/evidence/workspaces/gate-a1-task-environments")


class GateA1ConfigurationError(ValueError):
    """Raised when the frozen pilot cannot define a deterministic acquisition set."""


@dataclass(frozen=True)
class DevelopmentSubsetExclusion:
    task_id: str
    reason: str


@dataclass(frozen=True)
class DevelopmentSubset:
    schema_version: int
    selection_revision: int
    scope: str
    supersedes: str
    scope_revision_rationale: str
    acquisition_executed_before_revision: bool
    retrieval_results_existed_before_revision: bool
    included_acquisition_tasks: tuple[str, ...]
    included_transfer_tasks: tuple[str, ...]
    excluded_tasks: tuple[DevelopmentSubsetExclusion, ...]

    @property
    def excluded_acquisition_tasks(self) -> tuple[DevelopmentSubsetExclusion, ...]:
        acquisition = set(CANONICAL_ACQUISITION_TASK_IDS)
        return tuple(item for item in self.excluded_tasks if item.task_id in acquisition)

    @property
    def excluded_transfer_tasks(self) -> tuple[DevelopmentSubsetExclusion, ...]:
        transfer = set(CANONICAL_TRANSFER_TASK_IDS)
        return tuple(item for item in self.excluded_tasks if item.task_id in transfer)


@dataclass(frozen=True)
class FrozenBaseline:
    task_id: str
    repository: str
    commit: str


@dataclass(frozen=True)
class PilotRecord:
    task_id: str
    repository: str
    image_name: str
    family_id: str
    occurrence_index: int
    chronological_index: int


class GateA1RetrievalAudit(BaseModel):
    """One shadow-retrieval decision with no evaluator family metadata."""

    model_config = ConfigDict(extra="forbid")

    current_task_id: str
    current_chronological_index: int
    tool: str
    operation: str
    retrieval_query_version: str
    vector_candidate_count: int = Field(ge=0)
    selected_pattern_id: str | None = None
    selected_pattern_source_task_id: str | None = None
    selected_pattern_source_chronological_index: int | None = None
    vector_score: float | None = None
    eligible_candidate_count: int = Field(ge=0)
    rejection_reasons: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    no_selection_reason: str | None = None
    retrieval_latency_ms: float = Field(ge=0)


class ComponentRetrievalEvaluator:
    """Evaluate V2 retrieval without advice delivery or ModelRetry."""

    def __init__(self, retrieval: RecoveryPatternRetrievalService) -> None:
        self._retrieval = retrieval

    def evaluate(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
    ) -> GateA1RetrievalAudit:
        if planned_action.task_id != task.id:
            raise ValueError("planned action task_id must match evaluation task")
        started = perf_counter()
        result = self._retrieval.retrieve(task, planned_action, environment)
        elapsed_ms = (perf_counter() - started) * 1000
        selected = result.selected_pattern
        return GateA1RetrievalAudit(
            current_task_id=task.id,
            current_chronological_index=task.chronological_index,
            tool=planned_action.tool,
            operation=planned_action.operation,
            retrieval_query_version=result.query_version,
            vector_candidate_count=len(result.candidates),
            selected_pattern_id=None if selected is None else selected.id,
            selected_pattern_source_task_id=(None if selected is None else selected.source_task_id),
            selected_pattern_source_chronological_index=(
                None if selected is None else selected.source_chronological_index
            ),
            vector_score=result.selected_vector_score,
            eligible_candidate_count=len(result.eligible_candidates),
            rejection_reasons={
                candidate.pattern_id: candidate.rejection_reasons
                for candidate in result.candidates
                if candidate.rejection_reasons
            },
            no_selection_reason=result.no_selection_reason,
            retrieval_latency_ms=elapsed_ms,
        )

    def evaluate_actions(
        self,
        task: Task,
        planned_actions: tuple[PlannedAction, ...],
        environment: EnvironmentContext,
    ) -> tuple[GateA1RetrievalAudit, ...]:
        """Evaluate a captured action sequence in order without intervention."""
        return tuple(self.evaluate(task, action, environment) for action in planned_actions)


@dataclass(frozen=True)
class AcquisitionIdentity:
    task_id: str
    repository: str
    upstream_image: str
    occurrence_index: int
    chronological_index: int


@dataclass(frozen=True)
class AcquisitionEnvironmentStatus:
    task_id: str
    repository: str
    upstream_image: str
    prepared_image: str
    environment_fingerprint: str
    validation_marker: str
    workspace: str
    docker_reachable: bool
    neo4j_reachable: bool
    neo4j_error: str
    ready: bool
    blockers: tuple[str, ...]


StatusChecker = Callable[[], tuple[bool, str]]
ImageChecker = Callable[[str], tuple[bool, str]]
BaselineChecker = Callable[[str, str], tuple[str, ...]]


def load_development_subset(project_root: Path) -> DevelopmentSubset:
    """Load the revised full Gate A1 development pilot and validate pilot.jsonl."""
    raw = _load_json(project_root / DEVELOPMENT_SUBSET_PATH)
    scope = raw.get("scope")
    if scope != "full_five_family_gate_a1_development_pilot":
        raise GateA1ConfigurationError("development subset scope is not the full Gate A1 pilot")
    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise GateA1ConfigurationError("Gate A1 development subset schema_version must be 1")
    selection_revision = raw.get("selection_revision")
    if selection_revision != 2:
        raise GateA1ConfigurationError("Gate A1 development subset selection_revision must be 2")
    supersedes = raw.get("supersedes")
    if supersedes != "resource_bounded_four_family_subset":
        raise GateA1ConfigurationError(
            "Gate A1 development subset must supersede the four-family selection"
        )
    rationale = raw.get("scope_revision_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise GateA1ConfigurationError("scope_revision_rationale must be a non-empty string")
    if raw.get("acquisition_executed_before_revision") is not False:
        raise GateA1ConfigurationError(
            "acquisition_executed_before_revision must be false"
        )
    if raw.get("retrieval_results_existed_before_revision") is not False:
        raise GateA1ConfigurationError(
            "retrieval_results_existed_before_revision must be false"
        )
    included_acquisition = _string_tuple(
        raw.get("included_acquisition_tasks"), "included_acquisition_tasks"
    )
    included_transfer = _string_tuple(raw.get("included_transfer_tasks"), "included_transfer_tasks")
    raw_excluded = raw.get("excluded_tasks")
    if not isinstance(raw_excluded, list):
        raise GateA1ConfigurationError("excluded_tasks must be a list")
    exclusions: list[DevelopmentSubsetExclusion] = []
    for value in cast(list[Any], raw_excluded):
        if not isinstance(value, dict):
            raise GateA1ConfigurationError("each excluded task must be an object")
        excluded = cast(dict[str, Any], value)
        task_id = excluded.get("task_id")
        reason = excluded.get("reason")
        if not isinstance(task_id, str) or not isinstance(reason, str) or not reason.strip():
            raise GateA1ConfigurationError(
                "each excluded task requires task_id and a non-empty reason"
            )
        exclusions.append(DevelopmentSubsetExclusion(task_id, reason))
    all_selected = (*included_acquisition, *included_transfer)
    all_excluded = tuple(item.task_id for item in exclusions)
    if len(set(all_selected)) != len(all_selected) or len(set(all_excluded)) != len(all_excluded):
        raise GateA1ConfigurationError("development subset contains duplicate task IDs")
    if set(all_selected) & set(all_excluded):
        raise GateA1ConfigurationError("included and excluded subset task IDs overlap")

    records = _pilot_records(project_root)
    by_task = {record.task_id: record for record in records}
    if len(by_task) != len(records):
        raise GateA1ConfigurationError("pilot manifest contains duplicate task IDs")
    _require_pilot_tasks(by_task, all_selected + all_excluded)
    if (
        tuple(
            sorted(included_acquisition, key=lambda task_id: by_task[task_id].chronological_index)
        )
        != included_acquisition
    ):
        raise GateA1ConfigurationError(
            "included acquisition tasks are not canonical chronological order"
        )
    acquisition_records = [by_task[task_id] for task_id in included_acquisition]
    if any(record.occurrence_index != 1 for record in acquisition_records):
        raise GateA1ConfigurationError("included acquisition tasks must be occurrence-1 tasks")
    if tuple(record.chronological_index for record in acquisition_records) != (1, 2, 3, 4, 5):
        raise GateA1ConfigurationError(
            "included acquisition chronology must remain canonical 1,2,3,4,5"
        )
    acquisition_families = {record.family_id for record in acquisition_records}
    for task_id in included_transfer:
        transfer = by_task[task_id]
        source = next(
            (
                record
                for record in records
                if record.family_id == transfer.family_id and record.occurrence_index == 1
            ),
            None,
        )
        if source is None or source.task_id not in included_acquisition:
            raise GateA1ConfigurationError(
                f"included transfer {task_id} has no included occurrence-1 acquisition family"
            )
        if transfer.family_id not in acquisition_families:
            raise GateA1ConfigurationError(f"included transfer family is not acquired: {task_id}")
    for task_id in CANONICAL_ACQUISITION_TASK_IDS + CANONICAL_TRANSFER_TASK_IDS:
        if task_id not in by_task:
            raise GateA1ConfigurationError(f"canonical pilot task is missing: {task_id}")
    if exclusions:
        raise GateA1ConfigurationError("full Gate A1 development pilot cannot exclude tasks")
    if included_acquisition != CANONICAL_ACQUISITION_TASK_IDS:
        raise GateA1ConfigurationError("full Gate A1 acquisition sequence is not canonical")
    if included_transfer != CANONICAL_TRANSFER_TASK_IDS:
        raise GateA1ConfigurationError("full Gate A1 transfer sequence is not canonical")
    return DevelopmentSubset(
        schema_version=1,
        selection_revision=2,
        scope=cast(str, scope),
        supersedes=cast(str, supersedes),
        scope_revision_rationale=rationale,
        acquisition_executed_before_revision=False,
        retrieval_results_existed_before_revision=False,
        included_acquisition_tasks=included_acquisition,
        included_transfer_tasks=included_transfer,
        excluded_tasks=tuple(exclusions),
    )


def load_frozen_baselines(project_root: Path) -> tuple[FrozenBaseline, ...]:
    """Load the five clean local acquisition snapshots without mutating them."""
    raw = _load_json(project_root / BASELINE_MANIFEST_PATH)
    values = raw.get("repositories")
    if not isinstance(values, list):
        raise GateA1ConfigurationError("repositories must be a list")
    baselines: list[FrozenBaseline] = []
    for value in cast(list[Any], values):
        if not isinstance(value, dict):
            raise GateA1ConfigurationError("each baseline entry must be an object")
        baseline = cast(dict[str, Any], value)
        task_id = baseline.get("task_id")
        repository = baseline.get("repository")
        commit = baseline.get("commit")
        if not all(
            isinstance(item, str) and item.strip() for item in (task_id, repository, commit)
        ):
            raise GateA1ConfigurationError(
                "baseline entries require task_id, repository, and commit"
            )
        baselines.append(
            FrozenBaseline(cast(str, task_id), cast(str, repository), cast(str, commit))
        )
    return tuple(baselines)


def validate_frozen_baselines(
    project_root: Path,
    *,
    subset: DevelopmentSubset | None = None,
    baseline_checker: BaselineChecker | None = None,
) -> dict[str, tuple[str, ...]]:
    """Validate required clean Git snapshots without changing repository state."""
    selected = subset or load_development_subset(project_root)
    baselines = load_frozen_baselines(project_root)
    by_task = {baseline.task_id: baseline for baseline in baselines}
    if len(baselines) != len(selected.included_acquisition_tasks) or set(by_task) != set(
        selected.included_acquisition_tasks
    ):
        raise GateA1ConfigurationError(
            "Gate A1 baseline manifest must contain exactly the included acquisition tasks"
        )
    pilot_by_task = {
        identity.task_id: identity
        for identity in load_acquisition_identities(
            project_root, task_ids=selected.included_acquisition_tasks
        )
    }
    repository_mismatches = [
        task_id
        for task_id, baseline in by_task.items()
        if baseline.repository != pilot_by_task[task_id].repository
    ]
    if repository_mismatches:
        raise GateA1ConfigurationError(
            "baseline repository does not match pilot identity for: "
            + ", ".join(sorted(repository_mismatches))
        )

    def default_checker(repository: str, commit: str) -> tuple[str, ...]:
        return _check_baseline(project_root, repository, commit)

    checker: BaselineChecker = baseline_checker or default_checker
    return {
        task_id: checker(by_task[task_id].repository, by_task[task_id].commit)
        for task_id in selected.included_acquisition_tasks
    }


def _check_baseline(
    project_root: Path,
    repository: str,
    expected_commit: str,
) -> tuple[str, ...]:
    path = project_root / "benchmark/workspaces" / repository
    if not path.is_dir():
        return (f"baseline repository directory is missing: {path.as_posix()}",)
    blockers: list[str] = []
    inside = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return (f"baseline is not a Git repository: {path.as_posix()}",)
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
    )
    if head.returncode != 0 or head.stdout.strip() != expected_commit:
        blockers.append(
            f"baseline HEAD mismatch: {head.stdout.strip() or '<unavailable>'} != {expected_commit}"
        )
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--short"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
    )
    if status.returncode != 0:
        blockers.append("baseline Git status could not be read")
    elif status.stdout.strip():
        blockers.append("baseline Git worktree is dirty")
    return tuple(blockers)


def _pilot_records(project_root: Path) -> tuple[PilotRecord, ...]:
    records: list[PilotRecord] = []
    for record in _load_jsonl(project_root / "benchmark/manifests/pilot.jsonl"):
        values = (
            record.get("task_id"),
            record.get("repository"),
            record.get("image_name"),
            record.get("family_id"),
            record.get("occurrence_index"),
            record.get("chronological_index"),
        )
        if (
            not isinstance(values[0], str)
            or not isinstance(values[1], str)
            or not isinstance(values[2], str)
            or not isinstance(values[3], str)
            or not isinstance(values[4], int)
            or not isinstance(values[5], int)
        ):
            raise GateA1ConfigurationError("pilot record has invalid identity metadata")
        records.append(PilotRecord(*cast(tuple[Any, ...], values)))
    return tuple(records)


def _require_pilot_tasks(by_task: dict[str, PilotRecord], task_ids: tuple[str, ...]) -> None:
    missing = [task_id for task_id in task_ids if task_id not in by_task]
    if missing:
        raise GateA1ConfigurationError(f"subset task IDs missing from pilot manifest: {missing}")


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise GateA1ConfigurationError(f"{field} must be a list of non-empty strings")
    items = cast(list[Any], value)
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise GateA1ConfigurationError(f"{field} must be a list of non-empty strings")
    return tuple(cast(str, item) for item in items)


def load_acquisition_identities(
    project_root: Path,
    *,
    task_ids: tuple[str, ...] | None = None,
) -> tuple[AcquisitionIdentity, ...]:
    """Load and validate occurrence-1 acquisition identities from the pilot only."""
    subset = load_development_subset(project_root)
    selected_task_ids = task_ids or subset.included_acquisition_tasks
    if set(selected_task_ids) != set(subset.included_acquisition_tasks):
        raise GateA1ConfigurationError(
            "Gate A1 acquisition identities must match the frozen development subset"
        )
    records = _load_jsonl(project_root / "benchmark/manifests/pilot.jsonl")
    wanted = set(selected_task_ids)
    selected = [record for record in records if record.get("task_id") in wanted]
    if {record.get("task_id") for record in selected} != wanted:
        raise GateA1ConfigurationError(
            "pilot manifest is missing one or more Gate A1 acquisition task identities"
        )
    ordered = sorted(selected, key=lambda record: int(record.get("chronological_index", -1)))
    if tuple(str(record.get("task_id")) for record in ordered) != selected_task_ids:
        raise GateA1ConfigurationError(
            "Gate A1 acquisition task IDs must be in pilot chronological order"
        )
    identities: list[AcquisitionIdentity] = []
    expected_chronology = tuple(int(record["chronological_index"]) for record in ordered)
    for record in ordered:
        occurrence = record.get("occurrence_index")
        chronological = record.get("chronological_index")
        if (
            not isinstance(occurrence, int)
            or not isinstance(chronological, int)
            or occurrence != 1
            or chronological not in expected_chronology
        ):
            raise GateA1ConfigurationError(
                "Gate A1 acquisition identities must be occurrence-1 tasks with "
                "chronological indexes must remain canonical"
            )
        repository = record.get("repository")
        upstream_image = record.get("image_name")
        if not isinstance(repository, str) or not repository.strip():
            raise GateA1ConfigurationError(f"pilot identity lacks repository for {record}")
        if not isinstance(upstream_image, str) or not upstream_image.strip():
            raise GateA1ConfigurationError(f"pilot identity lacks image_name for {record}")
        identities.append(
            AcquisitionIdentity(
                task_id=str(record["task_id"]),
                repository=repository,
                upstream_image=upstream_image,
                occurrence_index=int(occurrence),
                chronological_index=int(chronological),
            )
        )
    return tuple(identities)


def assess_acquisition_environments(
    project_root: Path,
    *,
    task_ids: tuple[str, ...] | None = None,
    docker_status_checker: StatusChecker | None = None,
    docker_image_checker: ImageChecker | None = None,
    neo4j_status_checker: StatusChecker | None = None,
    baseline_checker: BaselineChecker | None = None,
) -> tuple[AcquisitionEnvironmentStatus, ...]:
    """Check the separate Gate A1 contract before any acquisition/provider call."""
    subset = load_development_subset(project_root)
    selected_task_ids = task_ids or subset.included_acquisition_tasks
    identities = load_acquisition_identities(project_root, task_ids=selected_task_ids)
    try:
        baseline_blockers = validate_frozen_baselines(
            project_root,
            subset=subset,
            baseline_checker=baseline_checker,
        )
    except GateA1ConfigurationError as error:
        baseline_blockers = {
            task_id: (f"baseline configuration is invalid: {error}",)
            for task_id in selected_task_ids
        }
    contract_path = project_root / ACQUISITION_CONTRACT
    contract = _load_json(contract_path) if contract_path.is_file() else {}
    contract_records = contract.get("environments", [])
    if not isinstance(contract_records, list):
        contract_records = []
    by_task: dict[str, dict[str, Any]] = {}
    for value in cast(list[Any], contract_records):
        if isinstance(value, dict) and "task_id" in value:
            record = cast(dict[str, Any], value)
            by_task[str(record["task_id"])] = record
    docker_ready, docker_error = (docker_status_checker or _docker_daemon_status)()
    image_checker = docker_image_checker or _docker_image_status
    neo4j_ready, neo4j_error = (neo4j_status_checker or _neo4j_endpoint_status)()
    statuses: list[AcquisitionEnvironmentStatus] = []
    for identity in identities:
        entry = by_task.get(identity.task_id, {})
        blockers: list[str] = []
        blockers.extend(baseline_blockers.get(identity.task_id, ()))
        if not entry:
            blockers.append("task has no separate Gate A1 prepared environment entry")
        upstream_image = str(entry.get("upstream_image", ""))
        if upstream_image != identity.upstream_image:
            blockers.append(
                "prepared environment upstream image does not match pilot identity: "
                f"{upstream_image or '<missing>'} != {identity.upstream_image}"
            )
        prepared_image = str(entry.get("prepared_image", ""))
        fingerprint = str(entry.get("environment_fingerprint", ""))
        marker_value = str(entry.get("validation_marker", ""))
        workspace_value = str(entry.get("workspace", ""))
        if not prepared_image:
            blockers.append("prepared image is missing from the Gate A1 contract")
        if not entry.get("upstream_digest"):
            blockers.append("upstream image digest is missing from the Gate A1 contract")
        if not fingerprint:
            blockers.append("environment fingerprint is missing")
        if not marker_value:
            blockers.append("validation marker is missing")
        marker = _project_path(project_root, marker_value)
        workspace = _project_path(project_root, workspace_value)
        if not marker.is_file():
            blockers.append(f"validation marker is missing: {marker.as_posix()}")
        else:
            metadata = _read_object(marker)
            if not isinstance(metadata, dict):
                blockers.append("validation marker does not say validated")
            else:
                marker_metadata = cast(dict[str, Any], metadata)
                if marker_metadata.get("validated") is not True:
                    blockers.append("validation marker does not say validated")
                elif marker_metadata.get("environment_fingerprint") != fingerprint:
                    blockers.append("validation marker fingerprint does not match contract")
        if not workspace.is_dir():
            blockers.append(f"workspace is missing: {workspace.as_posix()}")
        if not docker_ready:
            blockers.append(docker_error)
        elif prepared_image:
            image_exists, image_error = image_checker(prepared_image)
            if not image_exists:
                blockers.append(image_error)
        if not neo4j_ready:
            blockers.append(f"neo4j_unreachable: {neo4j_error}")
        statuses.append(
            AcquisitionEnvironmentStatus(
                task_id=identity.task_id,
                repository=identity.repository,
                upstream_image=identity.upstream_image,
                prepared_image=prepared_image,
                environment_fingerprint=fingerprint,
                validation_marker=marker_value,
                workspace=workspace.as_posix(),
                docker_reachable=docker_ready,
                neo4j_reachable=neo4j_ready,
                neo4j_error=neo4j_error,
                ready=not blockers,
                blockers=tuple(blockers),
            )
        )
    return tuple(statuses)


def prepare_gate_a1_environments(
    project_root: Path,
    *,
    baseline_root: Path | None = None,
    execution_root: Path | None = None,
    artifact_root: Path | None = None,
) -> Path:
    """Prepare only the frozen full five-family acquisition tasks."""
    subset = load_development_subset(project_root)
    identities = load_acquisition_identities(
        project_root, task_ids=subset.included_acquisition_tasks
    )
    baseline_blockers = validate_frozen_baselines(project_root, subset=subset)
    if any(baseline_blockers.values()):
        raise RuntimeError(f"Gate A1 baselines are not reproducible: {baseline_blockers}")
    baseline = (baseline_root or (project_root / "benchmark/workspaces")).resolve()
    execution = (execution_root or (project_root / ACQUISITION_ENVIRONMENT_ROOT)).resolve()
    artifacts = (artifact_root or (project_root / "research/evidence/results")).resolve()
    from experiments.sprint3 import run_prepare

    preparation_result = run_prepare(
        project_root=project_root,
        baseline_root=baseline,
        execution_root=execution.parent,
        artifact_root=artifacts,
        task_ids=subset.included_acquisition_tasks,
        policy_path=project_root / ACQUISITION_POLICY,
        environment_subdirectory=execution.name,
        result_subdirectory="gate_a1",
    )
    result = _load_json(preparation_result)
    task_results = {
        str(task["task_id"]): task for task in cast(list[dict[str, Any]], result.get("tasks", []))
    }
    entries: list[dict[str, Any]] = []
    for identity in identities:
        prepared = task_results[identity.task_id]
        marker = Path(str(prepared["validation_marker"])).resolve()
        metadata = _read_object(marker)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"invalid environment marker for {identity.task_id}: {marker}")
        marker_metadata = cast(dict[str, Any], metadata)
        metadata = marker_metadata
        prepared_image = str(prepared["container_image"])
        image_ready, image_error = _docker_image_status(prepared_image)
        if not image_ready:
            raise RuntimeError(image_error)
        workspace = execution / identity.task_id / "workspace"
        source = baseline / identity.repository
        if not source.is_dir():
            raise RuntimeError(f"missing clean baseline workspace: {source}")
        shutil.copytree(source, workspace, dirs_exist_ok=False)
        _validate_prepared_runtime(
            prepared_image,
            workspace,
            str(prepared.get("container_python_executable") or "python"),
        )
        metadata.update({"validated": True, "gate_a1_runtime_preflight": True})
        marker.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
        entries.append(
            {
                "task_id": identity.task_id,
                "repository": identity.repository,
                "upstream_image": identity.upstream_image,
                "upstream_digest": str(
                    prepared.get("base_image_digest")
                    or marker_metadata.get("base_image_digest")
                    or ""
                ),
                "prepared_image": prepared_image,
                "environment_fingerprint": str(prepared["environment_fingerprint"]),
                "validation_marker": _relative_or_absolute(project_root, marker),
                "workspace": _relative_or_absolute(project_root, workspace),
                "runtime_type": "docker",
            }
        )
    contract_path = project_root / ACQUISITION_CONTRACT
    _write_json(
        contract_path,
        {
            "schema_version": 1,
            "gate": GATE_A1,
            "contract_type": "development_only_acquisition",
            "source_manifest": "benchmark/manifests/pilot.jsonl",
            "environment_root": _relative_or_absolute(project_root, execution),
            "environments": entries,
        },
    )
    return contract_path


def write_readiness_artifact(
    project_root: Path,
    statuses: tuple[AcquisitionEnvironmentStatus, ...],
) -> Path:
    """Write a non-destructive, timestamped Gate A1 readiness artifact."""
    subset = load_development_subset(project_root)
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = project_root / "research" / "evidence" / f"gs_e003_gate_a1_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    starting_commit = _git(project_root, "rev-parse", "HEAD")
    branch = _git(project_root, "branch", "--show-current")
    blockers = [
        {"task_id": status.task_id, "reasons": list(status.blockers)}
        for status in statuses
        if not status.ready
    ]
    status_name = "BLOCKED_ACQUISITION_ENVIRONMENT" if blockers else "READY"
    manifest = {
        "gate": GATE_A1,
        "created_at": now.isoformat(),
        "starting_commit": starting_commit,
        "branch": branch,
        "model_provider": "openrouter",
        "recovery_model": FROZEN_RECOVERY_MODEL,
        "abstraction_prompt_version": RECOVERY_ABSTRACTION_PROMPT_VERSION,
        "embedding_model": RECOVERY_PATTERN_EMBEDDING_MODEL,
        "embedding_dimension": RECOVERY_PATTERN_EMBEDDING_DIMENSION,
        "retrieval_query_version": RECOVERY_RETRIEVAL_QUERY_VERSION,
        "vector_index": VECTOR_INDEX,
        "development_scope": subset.scope,
        "selection_revision": subset.selection_revision,
        "supersedes": subset.supersedes,
        "scope_revision_rationale": subset.scope_revision_rationale,
        "acquisition_executed_before_scope_revision": (
            subset.acquisition_executed_before_revision
        ),
        "retrieval_results_existed_before_scope_revision": (
            subset.retrieval_results_existed_before_revision
        ),
        "canonical_acquisition_candidate_count": len(CANONICAL_ACQUISITION_TASK_IDS),
        "selected_local_acquisition_count": len(subset.included_acquisition_tasks),
        "excluded_acquisition_count": len(subset.excluded_acquisition_tasks),
        "acquisition_task_ids": [status.task_id for status in statuses],
        "included_acquisition_task_ids": list(subset.included_acquisition_tasks),
        "included_transfer_task_ids": list(subset.included_transfer_tasks),
        "excluded_acquisition_tasks": [asdict(item) for item in subset.excluded_acquisition_tasks],
        "excluded_transfer_tasks": [asdict(item) for item in subset.excluded_transfer_tasks],
        "subset_frozen_before_acquisition_execution": True,
        "scope_frozen_before_acquisition_execution": True,
        "frozen_baseline_manifest": BASELINE_MANIFEST_PATH.as_posix(),
        "frozen_baselines": [asdict(item) for item in load_frozen_baselines(project_root)],
        "acquisition_environment_contract": ACQUISITION_CONTRACT.as_posix(),
        "environment_statuses": [asdict(status) for status in statuses],
        "infrastructure_preflight": {
            "docker_reachable": all(status.docker_reachable for status in statuses),
            "neo4j_reachable": all(status.neo4j_reachable for status in statuses),
            "neo4j_errors": sorted(
                {status.neo4j_error for status in statuses if status.neo4j_error}
            ),
        },
        "blockers": blockers,
        "execution_started": False,
        "exact_command": "python -m graph_swarm.research.gate_a1 --project-root .",
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    _write_jsonl(
        artifact_dir / "acquisition.jsonl",
        [
            {
                "task_id": status.task_id,
                "repository": status.repository,
                "upstream_image": status.upstream_image,
                "prepared_image": status.prepared_image,
                "workspace": status.workspace,
                "status": "blocked" if not status.ready else "ready",
                "blockers": list(status.blockers),
            }
            for status in statuses
        ],
    )
    _write_jsonl(artifact_dir / "patterns.jsonl", [])
    _write_jsonl(artifact_dir / "retrieval_evaluations.jsonl", [])
    _write_jsonl(artifact_dir / "error_analysis.jsonl", [])
    _write_json(
        artifact_dir / "metrics.json",
        {
            "status": status_name,
            "development_scope": subset.scope,
            "selection_revision": subset.selection_revision,
            "canonical_acquisition_candidate_count": len(CANONICAL_ACQUISITION_TASK_IDS),
            "selected_local_acquisition_count": len(subset.included_acquisition_tasks),
            "excluded_acquisition_count": len(subset.excluded_acquisition_tasks),
            "scope_frozen_before_acquisition_execution": True,
            "acquisition_tasks_attempted": 0,
            "complete_trusted_recovery_lineages": 0,
            "patterns_generated": 0,
            "patterns_embedded": 0,
            "included_acquisition_tasks": len(subset.included_acquisition_tasks),
            "excluded_acquisition_tasks": len(subset.excluded_acquisition_tasks),
            "included_transfer_tasks": len(subset.included_transfer_tasks),
            "excluded_transfer_tasks": len(subset.excluded_transfer_tasks),
            "transfer_tasks_evaluated": 0,
            "transfer_task_denominator": len(subset.included_transfer_tasks),
            "actions_evaluated": 0,
            "selections": 0,
            "no_selection_cases": 0,
            "correct_top1": 0,
            "incorrect_top1": 0,
            "precision_at_1": None,
            "selection_coverage": None,
            "strict_correct_top1_all_labelled_opportunities": None,
            "candidate_recall_at_k": None,
            "median_retrieval_latency_ms": None,
            "p95_retrieval_latency_ms": None,
            "rejection_counts": {},
            "interpretation": (
                "Pattern acquisition quality was not evaluated because the environment "
                "preflight failed before execution."
            ),
            "blockers": blockers,
        },
    )
    readme_lines = [
        f"# {GATE_A1}",
        "",
        "Development-only readiness check. No acquisition, provider call, "
        "advice delivery, or behavioral T execution was performed.",
        "",
        f"Status: {status_name}",
        "",
        f"Development scope: {subset.scope}",
        f"Selection revision: {subset.selection_revision}",
        f"Supersedes: {subset.supersedes}",
        "Scope frozen before acquisition execution: true",
        "",
        "Pattern acquisition quality was not evaluated because the environment "
        "preflight failed before execution.",
    ]
    if subset.excluded_tasks:
        readme_lines.extend(
            [
                "",
                "Excluded from this local development diagnostic:",
                *[
                    f"- {item.task_id}: {item.reason}"
                    for item in subset.excluded_tasks
                ],
            ]
        )
    readme_lines.extend(
        [
            "",
            "## Blockers",
            "",
            *[f"- {item['task_id']}: {', '.join(item['reasons'])}" for item in blockers],
        ]
    )
    (artifact_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )
    return artifact_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the development-only Gate A1 readiness check")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--prepare-environments",
        action="store_true",
        help="prepare only the frozen Gate A1 acquisition tasks and write the separate contract",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if args.prepare_environments:
        print(prepare_gate_a1_environments(project_root))
        return 0
    statuses = assess_acquisition_environments(project_root)
    artifact_dir = write_readiness_artifact(project_root, statuses)
    if any(not status.ready for status in statuses):
        print(f"BLOCKED_ACQUISITION_ENVIRONMENT {artifact_dir}")
        return 2
    print(f"READY_FOR_GATE_A1_ACQUISITION {artifact_dir}")
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    if not isinstance(value, dict):
        raise GateA1ConfigurationError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GateA1ConfigurationError(f"could not read pilot manifest {path}: {error}") from error
    records: list[dict[str, Any]] = []
    for line in lines:
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(cast(dict[str, Any], value))
    return records


def _read_object(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _relative_or_absolute(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _git(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _docker_daemon_status() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "Docker CLI is unavailable"
    try:
        subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"Docker daemon is unavailable: {error}"
    return True, ""


def _docker_image_status(image: str) -> tuple[bool, str]:
    try:
        inspected = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"prepared image could not be inspected: {error}"
    if inspected.returncode != 0:
        detail = inspected.stderr.strip() or "docker image inspect failed"
        return False, f"prepared image is unavailable: {image}: {detail}"
    return True, ""


def _neo4j_endpoint_status() -> tuple[bool, str]:
    driver: Driver | None = None
    try:
        settings = get_settings()
        driver = cast(Any, GraphDatabase).driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        cast(Any, driver).verify_connectivity()
        return True, ""
    except Exception as error:  # driver errors vary by transport and DNS provider
        return False, str(error)
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass


def _validate_prepared_runtime(image: str, workspace: Path, python_executable: str) -> None:
    docker = shutil.which("docker") or "docker"
    completed = subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--volume",
            f"{workspace.resolve()}:/workspace:ro",
            "--workdir",
            "/workspace",
            image,
            python_executable,
            "--version",
        ],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"prepared runtime validation failed for {image}: {completed.stderr[-2000:]}"
        )


__all__ = [
    "AcquisitionEnvironmentStatus",
    "AcquisitionIdentity",
    "ComponentRetrievalEvaluator",
    "DevelopmentSubset",
    "DevelopmentSubsetExclusion",
    "FrozenBaseline",
    "GateA1ConfigurationError",
    "GateA1RetrievalAudit",
    "assess_acquisition_environments",
    "load_acquisition_identities",
    "load_development_subset",
    "load_frozen_baselines",
    "prepare_gate_a1_environments",
    "validate_frozen_baselines",
    "write_readiness_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
