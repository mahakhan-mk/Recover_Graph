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

    audit = ComponentRetrievalEvaluator(
        cast(RecoveryPatternRetrievalService, retrieval)
    ).evaluate(task, action, environment)

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


def _test_project(tmp_path: Path, source_root: Path, *, complete: bool = True) -> Path:
    (tmp_path / "benchmark" / "manifests").mkdir(parents=True)
    shutil.copy(
        source_root / "benchmark/manifests/pilot.jsonl",
        tmp_path / "benchmark/manifests/pilot.jsonl",
    )
    (tmp_path / "configs" / "research").mkdir(parents=True)
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

    return {
        "docker_status_checker": docker_status,
        "docker_image_checker": image_status,
        "neo4j_status_checker": neo4j_status,
    }


def test_acquisition_identities_derive_from_pilot_in_occurrence_order(
    project_root: Path,
) -> None:
    identities = load_acquisition_identities(project_root)
    assert [identity.task_id for identity in identities] == [
        "GS-T001",
        "GS-T002",
        "GS-T003",
        "GS-T004",
        "GS-T005",
    ]
    assert [identity.upstream_image for identity in identities] == [
        "swebench/swesmith.x86_64.arrow-py_1776_arrow.1d70d009",
        "swebench/swesmith.x86_64.pygments_1776_pygments.27649ebb",
        "swebench/swesmith.x86_64.sunpy_1776_sunpy.f8edfd5c",
        "swebench/swesmith.x86_64.project-monai_1776_monai.a09c1f08",
        "swebench/swesmith.x86_64.cknd_1776_stackprinter.219fcc52",
    ]
    assert all(identity.occurrence_index == 1 for identity in identities)


def test_readiness_uses_separate_contract_and_leaves_frozen_b1_manifest_unchanged(
    project_root: Path, tmp_path: Path
) -> None:
    b1_manifest = project_root / "configs/research/docker_environments.json"
    before = hashlib.sha256(b1_manifest.read_bytes()).digest()
    statuses = assess_acquisition_environments(
        _test_project(tmp_path, project_root), **_ready_checks()
    )
    after = hashlib.sha256(b1_manifest.read_bytes()).digest()
    assert before == after
    assert all(status.ready for status in statuses)


def test_mismatched_upstream_image_fails_readiness(
    project_root: Path, tmp_path: Path
) -> None:
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
    assert (
        "Pattern acquisition quality was not evaluated because the environment "
        "preflight failed before execution."
        in (artifact / "README.md").read_text()
    )
