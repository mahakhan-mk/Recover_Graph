import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

import pytest
from _pytest.monkeypatch import MonkeyPatch

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.tasks import Task
from graph_swarm.research import gate_a1
from graph_swarm.research.gate_a1 import (
    ComponentRetrievalEvaluator,
    assess_acquisition_environments,
    load_acquisition_identities,
    load_development_subset,
    prepare_gate_a1_environments,
    validate_frozen_baselines,
    write_readiness_artifact,
)
from graph_swarm.retrieval.service import (
    RecoveryCandidateEvaluation,
    RecoveryPatternRetrievalService,
    RecoveryRetrievalResult,
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class FakeRetrieval:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
    ) -> RecoveryRetrievalResult:
        self.calls += 1
        return RecoveryRetrievalResult(
            query_version="v1",
            query_text="Task: safe query",
            candidates=(
                RecoveryCandidateEvaluation(
                    pattern_id="pattern-001",
                    vector_score=0.9,
                    eligible=True,
                ),
            ),
            eligible_candidates=(
                RecoveryCandidateEvaluation(
                    pattern_id="pattern-001",
                    vector_score=0.9,
                    eligible=True,
                ),
            ),
            selected_pattern=None,
            selected_vector_score=None,
            no_selection_reason="shadow_evaluation_only",
        )


def test_component_evaluator_is_shadow_only_and_never_injects_advice() -> None:
    task = Task(
        id="task-001",
        problem_statement="Repair the implementation.",
        family_id="family-evaluator-only",
        repository="repo-current",
        chronological_index=6,
    )
    action = PlannedAction(
        id="action-001",
        run_id="run-001",
        task_id=task.id,
        tool="write_file",
        operation="write_file",
        arguments={"path": "module.py", "content": "fixed"},
        planned_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    environment = EnvironmentContext(
        id="environment-001",
        repository=task.repository,
        runtime="python-3.13",
    )
    retrieval = FakeRetrieval()

    audit = ComponentRetrievalEvaluator(cast(RecoveryPatternRetrievalService, retrieval)).evaluate(
        task, action, environment
    )

    assert retrieval.calls == 1
    assert audit.current_task_id == task.id
    assert audit.current_chronological_index == task.chronological_index
    assert audit.selected_pattern_id is None
    assert audit.no_selection_reason == "shadow_evaluation_only"
    assert audit.retrieval_latency_ms >= 0
    assert "family_id" not in audit.model_dump()


class ReadinessChecks(TypedDict):
    docker_status_checker: gate_a1.StatusChecker
    docker_image_checker: gate_a1.ImageChecker
    neo4j_status_checker: gate_a1.StatusChecker
    baseline_checker: gate_a1.BaselineChecker


def _test_project(tmp_path: Path, source_root: Path, *, complete: bool = True) -> Path:
    (tmp_path / "benchmark" / "manifests").mkdir(parents=True)
    shutil.copy(
        source_root / "benchmark/manifests/pilot.jsonl",
        tmp_path / "benchmark/manifests/pilot.jsonl",
    )
    (tmp_path / "configs" / "research").mkdir(parents=True)
    shutil.copy(
        source_root / gate_a1.DEVELOPMENT_SUBSET_PATH,
        tmp_path / gate_a1.DEVELOPMENT_SUBSET_PATH,
    )
    shutil.copy(
        source_root / gate_a1.BASELINE_MANIFEST_PATH,
        tmp_path / gate_a1.BASELINE_MANIFEST_PATH,
    )
    identities = load_acquisition_identities(source_root)
    environments: list[dict[str, str]] = []
    for identity in identities:
        marker = tmp_path / "env" / identity.task_id / "environment.json"
        workspace = tmp_path / "env" / identity.task_id / "workspace"
        if complete:
            marker.parent.mkdir(parents=True, exist_ok=True)
            workspace.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "validated": True,
                        "environment_fingerprint": f"fingerprint-{identity.task_id}",
                    }
                ),
                encoding="utf-8",
            )
        environments.append(
            {
                "task_id": identity.task_id,
                "repository": identity.repository,
                "upstream_image": identity.upstream_image,
                "upstream_digest": "sha256:upstream",
                "prepared_image": f"prepared/{identity.task_id.lower()}:image",
                "environment_fingerprint": f"fingerprint-{identity.task_id}",
                "validation_marker": marker.relative_to(tmp_path).as_posix(),
                "workspace": workspace.relative_to(tmp_path).as_posix(),
            }
        )
    (tmp_path / gate_a1.ACQUISITION_CONTRACT).write_text(
        json.dumps({"environments": environments}), encoding="utf-8"
    )
    return tmp_path


def _ready_checks() -> ReadinessChecks:
    def docker_status() -> tuple[bool, str]:
        return True, ""

    def image_status(_image: str) -> tuple[bool, str]:
        return True, ""

    def neo4j_status() -> tuple[bool, str]:
        return True, ""

    def baseline_status(_repository: str, _commit: str) -> tuple[str, ...]:
        return ()

    return {
        "docker_status_checker": docker_status,
        "docker_image_checker": image_status,
        "neo4j_status_checker": neo4j_status,
        "baseline_checker": baseline_status,
    }


def test_acquisition_identities_derive_from_pilot_in_occurrence_order(
    project_root: Path,
) -> None:
    identities = load_acquisition_identities(project_root)
    assert [identity.task_id for identity in identities] == [
        "GS-T001",
        "GS-T002",
        "GS-T003",
        "GS-T005",
    ]
    assert [identity.upstream_image for identity in identities] == [
        "swebench/swesmith.x86_64.arrow-py_1776_arrow.1d70d009",
        "swebench/swesmith.x86_64.pygments_1776_pygments.27649ebb",
        "swebench/swesmith.x86_64.sunpy_1776_sunpy.f8edfd5c",
        "swebench/swesmith.x86_64.cknd_1776_stackprinter.219fcc52",
    ]
    assert all(identity.occurrence_index == 1 for identity in identities)
    assert [identity.chronological_index for identity in identities] == [1, 2, 3, 5]


def test_resource_bounded_subset_validates_canonical_transfer_families(
    project_root: Path,
) -> None:
    subset = load_development_subset(project_root)
    assert subset.included_acquisition_tasks == ("GS-T001", "GS-T002", "GS-T003", "GS-T005")
    assert subset.included_transfer_tasks == (
        "GS-T006",
        "GS-T007",
        "GS-T008",
        "GS-T010",
        "GS-T011",
        "GS-T012",
        "GS-T013",
        "GS-T015",
    )
    assert [item.task_id for item in subset.excluded_acquisition_tasks] == ["GS-T004"]
    assert [item.task_id for item in subset.excluded_transfer_tasks] == ["GS-T009", "GS-T014"]


def test_readiness_uses_separate_contract_and_leaves_frozen_b1_manifest_unchanged(
    project_root: Path, tmp_path: Path
) -> None:
    frozen_files = (
        project_root / "benchmark/manifests/pilot.jsonl",
        project_root / "configs/research/docker_environments.json",
        project_root / "configs/research/benchmark_environments.toml",
        project_root / "configs/experiments/gate_b1.yaml",
    )
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in frozen_files}
    statuses = assess_acquisition_environments(
        _test_project(tmp_path, project_root), **_ready_checks()
    )
    after = {path: hashlib.sha256(path.read_bytes()).digest() for path in frozen_files}
    assert before == after
    assert all(status.ready for status in statuses)
    assert [status.task_id for status in statuses] == ["GS-T001", "GS-T002", "GS-T003", "GS-T005"]


def test_mismatched_upstream_image_fails_readiness(project_root: Path, tmp_path: Path) -> None:
    project = _test_project(tmp_path, project_root)
    contract = json.loads((project / gate_a1.ACQUISITION_CONTRACT).read_text())
    contract["environments"][0]["upstream_image"] = "wrong/image"
    (project / gate_a1.ACQUISITION_CONTRACT).write_text(json.dumps(contract))
    statuses = assess_acquisition_environments(project, **_ready_checks())
    assert any("does not match pilot identity" in reason for reason in statuses[0].blockers)


def test_missing_workspace_fails_readiness(project_root: Path, tmp_path: Path) -> None:
    project = _test_project(tmp_path, project_root)
    workspace = project / "env/GS-T001/workspace"
    workspace.rmdir()
    statuses = assess_acquisition_environments(project, **_ready_checks())
    assert any("workspace is missing" in reason for reason in statuses[0].blockers)


def test_docker_unavailable_fails_readiness(project_root: Path, tmp_path: Path) -> None:
    project = _test_project(tmp_path, project_root)
    statuses = assess_acquisition_environments(
        project,
        docker_status_checker=lambda: (False, "Docker daemon is unavailable"),
        docker_image_checker=lambda _image: (_ for _ in ()).throw(AssertionError()),
        neo4j_status_checker=lambda: (True, ""),
    )
    assert all(not status.ready for status in statuses)
    assert any("Docker daemon is unavailable" in reason for reason in statuses[0].blockers)


def test_neo4j_unreachable_fails_without_database_writes(
    project_root: Path, tmp_path: Path
) -> None:
    project = _test_project(tmp_path, project_root)
    checks = _ready_checks()
    statuses = assess_acquisition_environments(
        project,
        docker_status_checker=checks["docker_status_checker"],
        docker_image_checker=checks["docker_image_checker"],
        neo4j_status_checker=lambda: (
            False,
            "Failed to DNS resolve address example.databases.neo4j.io:7687",
        ),
    )
    assert all(not status.ready for status in statuses)
    assert any("neo4j_unreachable" in reason for reason in statuses[0].blockers)


def test_frozen_baseline_sha_mismatch_fails_closed(project_root: Path) -> None:
    blockers = validate_frozen_baselines(
        project_root,
        baseline_checker=lambda repository, commit: (
            ("baseline HEAD mismatch",) if repository.endswith("arrow.1d70d009") else ()
        ),
    )
    assert blockers["GS-T001"] == ("baseline HEAD mismatch",)
    assert blockers["GS-T002"] == ()


def test_dirty_acquisition_baseline_fails_closed(project_root: Path) -> None:
    blockers = validate_frozen_baselines(
        project_root,
        baseline_checker=lambda repository, commit: (
            ("baseline Git worktree is dirty",)
            if repository.endswith("stackprinter.219fcc52")
            else ()
        ),
    )
    assert blockers["GS-T005"] == ("baseline Git worktree is dirty",)


def test_gate_a1_preparation_never_requests_excluded_monai_task(
    project_root: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    project = _test_project(tmp_path, project_root)
    identities = load_acquisition_identities(project)
    for identity in identities:
        (project / "benchmark" / "workspaces" / identity.repository).mkdir(
            parents=True, exist_ok=True
        )
    calls: list[tuple[str, ...]] = []

    def fake_run_prepare(**kwargs: object) -> Path:
        task_ids = tuple(cast(tuple[str, ...], kwargs["task_ids"]))
        calls.append(task_ids)
        execution_root = cast(Path, kwargs["execution_root"])
        subdirectory = cast(str, kwargs["environment_subdirectory"])
        environment_root = execution_root / subdirectory
        result_root = cast(Path, kwargs["artifact_root"]) / "GS-E003" / "gate_a1"
        result_root.mkdir(parents=True, exist_ok=True)
        tasks: list[dict[str, str]] = []
        for task_id in task_ids:
            marker = environment_root / task_id / "fingerprint" / "environment.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "validated": False,
                        "environment_fingerprint": "fingerprint",
                        "base_image_digest": "sha256:base",
                    }
                ),
                encoding="utf-8",
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "container_image": f"prepared/{task_id.lower()}:image",
                    "environment_fingerprint": "fingerprint",
                    "validation_marker": str(marker),
                    "base_image_digest": "sha256:base",
                    "container_python_executable": "python",
                }
            )
        result = result_root / "preparation-test.json"
        result.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
        return result

    monkeypatch.setattr("experiments.sprint3.run_prepare", fake_run_prepare)

    def image_status(_image: str) -> tuple[bool, str]:
        return True, ""

    def validate_runtime(*_args: object) -> None:
        return None

    def validate_baselines(*_args: object, **_kwargs: object) -> dict[str, tuple[str, ...]]:
        return {task_id: () for task_id in ("GS-T001", "GS-T002", "GS-T003", "GS-T005")}

    monkeypatch.setattr(gate_a1, "_docker_image_status", image_status)
    monkeypatch.setattr(gate_a1, "_validate_prepared_runtime", validate_runtime)
    monkeypatch.setattr(
        gate_a1,
        "validate_frozen_baselines",
        validate_baselines,
    )
    contract_path = prepare_gate_a1_environments(project)
    contract = json.loads(contract_path.read_text())
    assert calls == [("GS-T001", "GS-T002", "GS-T003", "GS-T005")]
    assert [entry["task_id"] for entry in contract["environments"]] == list(calls[0])
    assert "project-monai" not in contract_path.read_text()


def test_zero_attempted_is_classified_as_environment_blocked(
    project_root: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    project = _test_project(tmp_path, project_root, complete=False)
    statuses = assess_acquisition_environments(
        project,
        docker_status_checker=lambda: (False, "Docker daemon is unavailable"),
        neo4j_status_checker=lambda: (False, "neo4j unavailable"),
    )

    def fake_git(*_args: object) -> str:
        return "test"

    monkeypatch.setattr(gate_a1, "_git", fake_git)
    artifact = write_readiness_artifact(project, statuses)
    metrics = json.loads((artifact / "metrics.json").read_text())
    assert metrics["status"] == "BLOCKED_ACQUISITION_ENVIRONMENT"
    assert metrics["acquisition_tasks_attempted"] == 0
    assert metrics["complete_trusted_recovery_lineages"] == 0
    assert metrics["patterns_generated"] == 0
    assert metrics["patterns_embedded"] == 0
    assert metrics["included_acquisition_tasks"] == 4
    assert metrics["excluded_acquisition_tasks"] == 1
    assert metrics["included_transfer_tasks"] == 8
    assert metrics["excluded_transfer_tasks"] == 2
    assert metrics["transfer_task_denominator"] == 8
    assert (
        "Pattern acquisition quality was not evaluated because the environment "
        "preflight failed before execution." in (artifact / "README.md").read_text()
    )
