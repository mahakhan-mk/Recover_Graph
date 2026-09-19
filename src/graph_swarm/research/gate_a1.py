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
ACQUISITION_TASK_IDS = tuple(f"GS-T{i:03d}" for i in range(1, 6))
TRANSFER_TASK_IDS = tuple(f"GS-T{i:03d}" for i in range(6, 16))
VECTOR_INDEX = "recovery_pattern_embedding_idx"
ACQUISITION_CONTRACT = Path("configs/research/gate_a1_docker_environments.json")
ACQUISITION_POLICY = Path("configs/research/gate_a1_benchmark_environments.toml")
ACQUISITION_ENVIRONMENT_ROOT = Path(
    "research/evidence/workspaces/gate-a1-task-environments"
)


class GateA1ConfigurationError(ValueError):
    """Raised when the frozen pilot cannot define a deterministic acquisition set."""


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
            selected_pattern_source_task_id=(
                None if selected is None else selected.source_task_id
            ),
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


def load_acquisition_identities(
    project_root: Path,
    *,
    task_ids: tuple[str, ...] = ACQUISITION_TASK_IDS,
) -> tuple[AcquisitionIdentity, ...]:
    """Load and validate occurrence-1 acquisition identities from the pilot only."""
    records = _load_jsonl(project_root / "benchmark/manifests/pilot.jsonl")
    wanted = set(task_ids)
    selected = [record for record in records if record.get("task_id") in wanted]
    if {record.get("task_id") for record in selected} != wanted:
        raise GateA1ConfigurationError(
            "pilot manifest is missing one or more Gate A1 acquisition task identities"
        )
    ordered = sorted(selected, key=lambda record: int(record.get("chronological_index", -1)))
    if tuple(str(record.get("task_id")) for record in ordered) != task_ids:
        raise GateA1ConfigurationError(
            "Gate A1 acquisition task IDs must be in pilot chronological order"
        )
    identities: list[AcquisitionIdentity] = []
    for expected_index, record in enumerate(ordered, start=1):
        occurrence = record.get("occurrence_index")
        chronological = record.get("chronological_index")
        if (
            not isinstance(occurrence, int)
            or not isinstance(chronological, int)
            or occurrence != 1
            or chronological != expected_index
        ):
            raise GateA1ConfigurationError(
                "Gate A1 acquisition identities must be occurrence-1 tasks with "
                "chronological indexes 1 through 5"
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
    task_ids: tuple[str, ...] = ACQUISITION_TASK_IDS,
    docker_status_checker: StatusChecker | None = None,
    docker_image_checker: ImageChecker | None = None,
    neo4j_status_checker: StatusChecker | None = None,
) -> tuple[AcquisitionEnvironmentStatus, ...]:
    """Check the separate Gate A1 contract before any acquisition/provider call."""
    identities = load_acquisition_identities(project_root, task_ids=task_ids)
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
    """Prepare only T001-T005 and write the isolated Gate A1 contract."""
    identities = load_acquisition_identities(project_root)
    baseline = (baseline_root or (project_root / "benchmark/workspaces")).resolve()
    execution = (
        execution_root or (project_root / ACQUISITION_ENVIRONMENT_ROOT)
    ).resolve()
    artifacts = (artifact_root or (project_root / "research/evidence/results")).resolve()
    from experiments.sprint3 import run_prepare

    preparation_result = run_prepare(
        project_root=project_root,
        baseline_root=baseline,
        execution_root=execution.parent,
        artifact_root=artifacts,
        task_ids=ACQUISITION_TASK_IDS,
        policy_path=project_root / ACQUISITION_POLICY,
        environment_subdirectory=execution.name,
        result_subdirectory="gate_a1",
    )
    result = _load_json(preparation_result)
    task_results = {
        str(task["task_id"]): task
        for task in cast(list[dict[str, Any]], result.get("tasks", []))
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
        "acquisition_task_ids": [status.task_id for status in statuses],
        "transfer_task_ids": list(TRANSFER_TASK_IDS),
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
            "acquisition_tasks_attempted": 0,
            "complete_trusted_recovery_lineages": 0,
            "patterns_generated": 0,
            "patterns_embedded": 0,
            "transfer_tasks_evaluated": 0,
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
    (artifact_dir / "README.md").write_text(
        "\n".join(
            (
                f"# {GATE_A1}",
                "",
                "Development-only readiness check. No acquisition, provider call, "
                "advice delivery, or behavioral T execution was performed.",
                "",
                f"Status: {status_name}",
                "",
                "Pattern acquisition quality was not evaluated because the environment "
                "preflight failed before execution.",
                "",
                "## Blockers",
                "",
                *(f"- {item['task_id']}: {', '.join(item['reasons'])}" for item in blockers),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the development-only Gate A1 readiness check")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--prepare-environments",
        action="store_true",
        help="prepare only T001-T005 and write the separate Gate A1 contract",
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
            f"prepared runtime validation failed for {image}: "
            f"{completed.stderr[-2000:]}"
        )


__all__ = [
    "ACQUISITION_TASK_IDS",
    "AcquisitionEnvironmentStatus",
    "AcquisitionIdentity",
    "ComponentRetrievalEvaluator",
    "GateA1ConfigurationError",
    "GateA1RetrievalAudit",
    "TRANSFER_TASK_IDS",
    "assess_acquisition_environments",
    "load_acquisition_identities",
    "prepare_gate_a1_environments",
    "write_readiness_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
