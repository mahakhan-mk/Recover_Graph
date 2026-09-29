from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import experiments.run_sprint3b_kilo_v3 as v3
from analysis.diagnose_sprint3b_kilo_v3_retrieval import budget_audit
from experiments.sprint3 import IsolatedTaskEnvironment
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import load_experiment_configuration
from graph_swarm.retrieval.applicability import RecoveryPatternApplicabilityService

# The B0 boundary test supplies intentionally minimal callable doubles.
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _prepared(
    *, runtime_type: str = "docker", python_version: str | None = "3.12.1"
) -> IsolatedTaskEnvironment:
    return IsolatedTaskEnvironment(
        task_id="GS-T006",
        python_executable=Path("docker"),
        python_version=python_version,
        runtime_type=runtime_type,
        container_image="graph-swarm/test:latest",
        container_python_executable="/opt/miniconda3/bin/python",
    )


def _context(*, runtime: str = "docker") -> EnvironmentContext:
    return EnvironmentContext(
        id="run-123-environment",
        repository="tornadoweb__tornado.d5ac65c1",
        runtime=runtime,
        versions={"python": "3.12.1"},
        markers={},
    )


def _pattern(pattern_id: str) -> RecoveryPattern:
    return RecoveryPattern(
        id=pattern_id,
        title="Edit recovery",
        guidance="Apply the observed edit recovery.",
        source_failure_id=f"{pattern_id}-failure",
        source_resolution_id=f"{pattern_id}-resolution",
        source_outcome_id=f"{pattern_id}-outcome",
        source_task_id="GS-T001",
        source_chronological_index=1,
        source_tool="run_command",
        source_operation="run_command",
        applicability_tool="edit_file",
        applicability_operation="edit_file",
        source_failure_type="test_failure",
        environment_constraints=EnvironmentConstraints(
            runtime="docker",
            versions={"python": "3.12.1"},
            markers={},
        ),
        verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
        evidence_count=1,
        evidence_summary="Observed successful recovery.",
        created_at=NOW,
    )


def _planned_edit() -> PlannedAction:
    return PlannedAction(
        id="call-edit",
        run_id="run-123",
        task_id="GS-T006",
        tool="edit_file",
        operation="edit_file",
        arguments={"path": "tornado/locale.py"},
        planned_at=NOW,
    )


def test_t_v3_environment_uses_prepared_docker_runtime() -> None:
    task = SimpleNamespace(repository="repo/current")
    context = v3.environment_context_for_prepared(task, "run-123", _prepared())
    assert context.runtime == "docker"


def test_t_v3_resolver_uses_same_prepared_environment_and_run_id() -> None:
    task = SimpleNamespace(id="GS-T006", repository="repo/current")
    resolver = v3.make_environment_resolver({"GS-T006": _prepared()})
    context = resolver(task, "run-specific")
    assert context.id == "run-specific-environment"
    assert context.runtime == "docker"
    assert context.repository == "repo/current"


def test_t_v3_environment_preserves_current_repository() -> None:
    task = SimpleNamespace(repository="repo/current")
    context = v3.environment_context_for_prepared(task, "run-123", _prepared())
    assert context.repository == "repo/current"


def test_t_v3_environment_propagates_prepared_python_version() -> None:
    task = SimpleNamespace(repository="repo/current")
    context = v3.environment_context_for_prepared(
        task, "run-123", _prepared(python_version="3.12.1")
    )
    assert context.versions == {"python": "3.12.1"}


def test_t_v3_environment_does_not_copy_acquisition_markers() -> None:
    task = SimpleNamespace(repository="repo/current")
    context = v3.environment_context_for_prepared(task, "run-123", _prepared())
    assert context.markers == {}
    assert "memory_write_only" not in context.markers
    assert "retrieval_performed" not in context.markers


def test_b0_rejects_treatment_runtime_or_environment_injection() -> None:
    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/sprint3b_kilo_v3.yaml", project_root=ROOT
    )
    with pytest.raises(v3.Sprint3BExecutionError, match="B0"):
        v3.build_condition_runner(
            configuration,
            condition=ExperimentCondition.B0,
            objective_evaluator=lambda *_args: True,
            recurrence_evaluator=lambda *_args: False,
            workspace_resolver=lambda *_args: ROOT,
            execution_runtime_resolver=lambda *_args: _prepared().agent_execution_runtime(),
            environment_resolver=lambda *_args: _context(),
        )


def test_t_environment_contains_only_applicability_relevant_transfer_facts() -> None:
    task = SimpleNamespace(repository="repo/current")
    dumped = v3.environment_context_for_prepared(task, "run-123", _prepared()).model_dump()
    assert set(dumped) == {"id", "repository", "runtime", "versions", "markers"}
    assert dumped["id"] == "run-123-environment"
    assert dumped["repository"] == "repo/current"
    assert dumped["runtime"] == "docker"
    assert dumped["versions"] == {"python": "3.12.1"}
    assert dumped["markers"] == {}


def test_v3_freeze_builder_is_deterministic_and_covers_replay_inputs(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/sprint3b_kilo_v3.yaml", project_root=ROOT
    )
    assert v3.PROTOCOL_REVISION == "sprint3b-kilo-reduced-b0-t-v3-budget28"
    assert v3.EXPECTED_MAX_ACTIONS == 28
    assert configuration.config.limits.max_actions == 28
    assert configuration.config.limits.max_actions == configuration.config.limits.max_actions

    freeze_path = tmp_path / "freeze.json"
    freeze = v3.create_freeze_artifact(
        project_root=ROOT,
        freeze_path=freeze_path,
    )
    assert freeze["status"] == "READY"
    assert freeze["protocol_revision"] == v3.PROTOCOL_REVISION
    assert freeze["config_version"] == "sprint3b-kilo-v3-budget28"
    assert freeze["limits"]["max_actions"] == 28
    hashes = freeze["freeze_input_hashes"]
    assert isinstance(hashes, dict)
    for relative in (
        "configs/experiments/sprint3b_kilo_v3.yaml",
        "src/graph_swarm/retrieval/applicability.py",
        "src/graph_swarm/retrieval/query.py",
        "src/graph_swarm/retrieval/service.py",
        "src/graph_swarm/domain/environment.py",
        "src/graph_swarm/domain/recovery_patterns.py",
    ):
        assert relative in hashes
    context = v3.load_frozen_execution_context(
        project_root=ROOT,
        freeze_path=freeze_path,
    )
    assert context.freeze["freeze_inputs_sha256"] == freeze["freeze_inputs_sha256"]

    changed_config = tmp_path / "sprint3b_kilo_v3_budget20.yaml"
    changed_config.write_text(
        (ROOT / "configs/experiments/sprint3b_kilo_v3.yaml")
        .read_text(encoding="utf-8")
        .replace("max_actions: 28", "max_actions: 20"),
        encoding="utf-8",
    )
    with pytest.raises(v3.FreezeValidationError, match="config hash"):
        v3.load_frozen_execution_context(
            project_root=ROOT,
            config_path=changed_config,
            freeze_path=freeze_path,
        )


def test_b0_and_t_share_the_same_budget28_configuration() -> None:
    configuration = load_experiment_configuration(
        ROOT / "configs/experiments/sprint3b_kilo_v3.yaml", project_root=ROOT
    )
    limits_by_condition = {
        ExperimentCondition.B0: configuration.config.limits.max_actions,
        ExperimentCondition.T: configuration.config.limits.max_actions,
    }
    assert limits_by_condition == {
        ExperimentCondition.B0: 28,
        ExperimentCondition.T: 28,
    }


def test_budget_audit_distinguishes_executed_actions_from_snapshot_plans() -> None:
    audit = budget_audit(
        ROOT / "research/evidence/results/GS-E003/sprint3b_kilo_v2/runs/GS-E003/T"
    )
    records = {
        (str(record["condition"]), str(record["task_id"])): record
        for record in cast(list[dict[str, object]], audit["records"])
    }
    t008 = records[("T", "GS-T008")]
    assert t008["configured_budget"] == 20
    assert t008["budget_consumed_count"] == 23
    assert t008["completed_tool_action_count"] == 19
    assert t008["budget_count_absent_from_artifact_events"] == 4
    assert t008["first_edit_file_action_index"] == 18
    assert t008["snapshot_planned_action_count"] == 25
    assert t008["final_observed_action_index"] == 19


def test_v2_evidence_paths_are_not_modified_by_v3_environment_construction() -> None:
    paths = (
        ROOT
        / "research/evidence/results/GS-E003/sprint3b_kilo_v2/posthoc_retrieval_diagnosis"
        / "diagnosis.json",
        ROOT
        / "research/evidence/results/GS-E003/sprint3b_kilo_v2/posthoc_retrieval_diagnosis"
        / "diagnosis.md",
    )
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    v3.environment_context_for_prepared(
        SimpleNamespace(repository="repo/current"), "run-123", _prepared()
    )
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before


@pytest.mark.parametrize(
    "pattern_id",
    [
        "recovery-pattern-ad07a6a45718b848a30ad377",
        "recovery-pattern-f490f62ab931191c6eac6db1",
    ],
)
def test_edit_file_candidates_are_not_rejected_for_matching_docker_runtime(pattern_id: str) -> None:
    decision = RecoveryPatternApplicabilityService().evaluate(
        _planned_edit(), _context(), _pattern(pattern_id)
    )
    assert decision.applicable
    assert "runtime_mismatch" not in decision.rejection_reasons


def test_genuine_runtime_mismatch_still_rejects_candidate() -> None:
    decision = RecoveryPatternApplicabilityService().evaluate(
        _planned_edit(), _context(runtime="python"), _pattern("pattern-runtime")
    )
    assert not decision.applicable
    assert "runtime_mismatch" in decision.rejection_reasons
