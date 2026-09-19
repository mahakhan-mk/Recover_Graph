from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import experiments.sprint3b as sprint3b
from analysis.gate_b1 import analyze_gate_b1, normalized_trajectory_signature
from experiments.sprint3b import (
    GATE_B1_CONDITIONS,
    GATE_B1_EXPECTED_RUNS,
    GATE_B1_MAX_ACTIONS,
    GATE_B1_MAX_REQUESTS,
    GATE_B1_NOMINAL_RPM,
    GATE_B1_PACING_SECONDS,
    GATE_B1_TASK_IDS,
    GATE_B1_TIMEOUT_SECONDS,
    GATE_B1_TOOL_RETRIES,
    AttemptRecord,
    AttemptStatus,
    GateB1DuplicateError,
    GateB1EvidenceStore,
    GateB1Manifest,
    GateB1ReplacementError,
    PlannedGateB1Execution,
    build_execution_plan,
    classify_attempt,
    execute_gate_b1,
    freeze_gate_b1,
    validate_artifact_contract,
)
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import (
    AdviceResult,
    ApplicabilityAssessment,
    RecoveryEvidence,
    RecoveryProvenance,
)
from graph_swarm.domain.resolutions import ResolutionStatus
from graph_swarm.research.contracts import (
    BoundedTermination,
    ExperimentCondition,
    ExperimentRunArtifact,
)
from graph_swarm.research.runner import ExperimentExecution


def _manifest(tmp_path: Path) -> GateB1Manifest:
    return freeze_gate_b1(
        project_root=Path("."),
        output_path=tmp_path / "freeze.json",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _attempt(
    manifest: GateB1Manifest,
    task_id: str,
    condition: ExperimentCondition,
    run_id: str,
    status: AttemptStatus,
) -> AttemptRecord:
    return AttemptRecord(
        manifest_id=manifest.manifest_id,
        experiment_id=manifest.experiment_id,
        task_id=task_id,
        condition=condition,
        run_id=run_id,
        environment_fingerprint=manifest.task_map[task_id].environment_fingerprint,
        actual_environment_fingerprint=manifest.task_map[task_id].environment_fingerprint,
        status=status,
        valid_observation=status.valid_observation,
    )


def _artifact(
    manifest: GateB1Manifest,
    task_id: str,
    condition: ExperimentCondition,
    run_id: str,
) -> ExperimentRunArtifact:
    task = manifest.task_map[task_id]
    action = PlannedAction(
        id=f"{run_id}-action",
        run_id=run_id,
        task_id=task_id,
        tool="agent",
        operation="task_completion",
        planned_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return ExperimentRunArtifact(
        experiment_id=manifest.experiment_id,
        gate_id="B1",
        execution_manifest_id=manifest.manifest_id,
        condition=condition,
        run_id=run_id,
        task_id=task_id,
        family_id=task.family_id,
        chronological_index=task.chronological_index,
        model="runtime-model",
        model_settings={"temperature": 0},
        prompt_version=manifest.prompt_version,
        planned_action=action,
        executed_action=ActionResult(
            action_id=action.id,
            tool_name="agent",
            success=True,
            started_at=action.planned_at,
            completed_at=action.planned_at,
        ),
        advice_received=(
            AdviceResult.no_advice("not applicable")
            if condition is ExperimentCondition.B0
            else AdviceResult.historical_recovery(
                matched_failure_episode_id=f"failure-{task_id}",
                matched_resolution_id=f"resolution-{task_id}",
                failed_tool="task_start",
                failed_operation="task_start",
                recovery_summary="frozen recovery pattern",
                resolution_status=ResolutionStatus.OBSERVED_SUCCESSFUL,
                recovery_evidence=RecoveryEvidence(
                    successful_observations=1,
                    failed_observations=0,
                ),
                applicability=ApplicabilityAssessment(
                    matched_fields=("frozen",),
                    repository=task.repository,
                    runtime="python",
                ),
                provenance=RecoveryProvenance(
                    failure_episode_id=f"failure-{task_id}",
                    resolution_id=f"resolution-{task_id}",
                    failed_action_id=action.id,
                    environment_id=f"environment-{task_id}",
                ),
            )
        ),
        advice_count=0 if condition is ExperimentCondition.B0 else 1,
        advice_review_id=None if condition is ExperimentCondition.B0 else task.review_id,
        advice_intervention_boundary=(
            None if condition is ExperimentCondition.B0 else "task_start"
        ),
        advice_delivery_timing=(
            None if condition is ExperimentCondition.B0 else "pre_first_model_request"
        ),
        task_success=True,
        known_failure_repeated=False,
        tool_calls=1,
        retries=0,
    )


def _populate_pairs(
    manifest: GateB1Manifest,
    evidence: Path,
    *,
    missing_artifact: bool = False,
    missing_raw: bool = False,
    wrong_o1_review: bool = False,
    environment_mismatch: bool = False,
    missing_o1_event: bool = False,
) -> GateB1EvidenceStore:
    store = GateB1EvidenceStore(evidence, manifest)
    for task_id in GATE_B1_TASK_IDS:
        for condition in GATE_B1_CONDITIONS:
            run_id = f"{condition.value}-{task_id}"
            artifact = _artifact(manifest, task_id, condition, run_id)
            if wrong_o1_review and condition is ExperimentCondition.O1:
                artifact = artifact.model_copy(update={"advice_review_id": "wrong-review"})
            artifact_path = evidence / f"{run_id}.artifact.json"
            raw_path = evidence / f"{run_id}.raw.json"
            if not missing_artifact:
                artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
            if not missing_raw:
                raw_path.write_text(
                    json.dumps(
                        {
                            "events": [],
                            "advice_events": (
                                []
                                if missing_o1_event
                                else [{}]
                                if condition is ExperimentCondition.O1
                                else []
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
            record = _attempt(
                manifest,
                task_id,
                condition,
                run_id,
                AttemptStatus.VALID_EXPERIMENTAL_RUN,
            ).model_copy(
                update={
                    "artifact_path": str(artifact_path),
                    "raw_evidence_path": str(raw_path),
                    "actual_environment_fingerprint": (
                        "wrong-environment"
                        if environment_mismatch
                        else manifest.task_map[task_id].environment_fingerprint
                    ),
                }
            )
            store.append(record)
    return store


def test_gate_b1_freeze_and_plan_are_exact_and_chronological(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest.task_ids == GATE_B1_TASK_IDS
    assert manifest.conditions == GATE_B1_CONDITIONS
    assert manifest.expected_runs == GATE_B1_EXPECTED_RUNS == 20

    plan = build_execution_plan(manifest)
    assert [item.task_id for item in plan[::2]] == list(GATE_B1_TASK_IDS)
    assert all(
        plan[index].condition is ExperimentCondition.B0
        and plan[index + 1].condition is ExperimentCondition.O1
        and plan[index].task_id == plan[index + 1].task_id
        for index in range(0, len(plan), 2)
    )


def test_cli_planning_does_not_enter_execution_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    def fail_execution(*_args: object, **_kwargs: object) -> object:
        pytest.fail("planning unexpectedly entered execution")

    monkeypatch.setattr(sprint3b, "execute_gate_b1_with_existing_stack", fail_execution)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sprint3b",
            "--manifest",
            str(manifest_path),
            "--task-id",
            "GS-T006",
        ],
    )
    sprint3b.main()
    assert '"condition": "B0"' in capsys.readouterr().out


def test_gate_b1_range_keeps_global_order_and_t_is_disabled(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    plan = build_execution_plan(manifest, start_task="GS-T008", end_task="GS-T010")
    assert [item.task_id for item in plan] == [
        "GS-T008",
        "GS-T008",
        "GS-T009",
        "GS-T009",
        "GS-T010",
        "GS-T010",
    ]
    with pytest.raises(ValueError, match="only B0 and O1"):
        build_execution_plan(manifest, conditions=(ExperimentCondition.T,))


def test_gate_b1_contract_constants_are_unchanged() -> None:
    assert GATE_B1_MAX_ACTIONS == 20
    assert GATE_B1_MAX_REQUESTS == 24
    assert GATE_B1_TOOL_RETRIES == 3
    assert GATE_B1_TIMEOUT_SECONDS == 300
    assert GATE_B1_PACING_SECONDS == 5
    assert GATE_B1_NOMINAL_RPM == 12


def test_invalid_attempts_are_immutable_and_manual_replacement_is_linked(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    store = GateB1EvidenceStore(tmp_path / "evidence", manifest)
    invalid = _attempt(
        manifest,
        "GS-T006",
        ExperimentCondition.B0,
        "invalid-b0",
        AttemptStatus.INVALID_PROVIDER_ATTEMPT,
    )
    store.append(invalid)
    with pytest.raises(GateB1ReplacementError):
        store.assert_can_attempt("GS-T006", ExperimentCondition.B0)
    valid = _attempt(
        manifest,
        "GS-T006",
        ExperimentCondition.B0,
        "valid-b0",
        AttemptStatus.VALID_EXPERIMENTAL_RUN,
    ).model_copy(update={"replacement_of": "invalid-b0"})
    store.append(valid, replacement_of="invalid-b0")
    assert [item.run_id for item in store.attempts()] == ["invalid-b0", "valid-b0"]
    with pytest.raises(GateB1DuplicateError):
        store.assert_can_attempt("GS-T006", ExperimentCondition.B0)


def test_bounded_termination_is_valid_and_provider_or_infrastructure_is_invalid(
    tmp_path: Path,
) -> None:
    bounded = _artifact(
        _manifest(tmp_path),
        "GS-T006",
        ExperimentCondition.B0,
        "bounded",
    ).model_copy(
        update={
            "termination": BoundedTermination(budget="tool_calls", configured_limit=20)
        }
    )
    assert classify_attempt(artifact=bounded, error=None) is AttemptStatus.BOUNDED_TERMINATION
    assert (
        classify_attempt(artifact=None, error=RuntimeError("provider 429"))
        is AttemptStatus.INVALID_PROVIDER_ATTEMPT
    )
    assert (
        classify_attempt(artifact=None, error=RuntimeError("prepared environment missing"))
        is AttemptStatus.INVALID_INFRASTRUCTURE_ATTEMPT
    )


def test_missing_o1_provenance_fails_artifact_validation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    artifact = _artifact(manifest, "GS-T006", ExperimentCondition.O1, "o1")
    with pytest.raises(ValueError, match="exact frozen Oracle"):
        validate_artifact_contract(
            artifact.model_copy(update={"advice_review_id": None}),
            manifest,
            expected_condition=ExperimentCondition.O1,
        )


def test_analysis_excludes_invalid_attempts_and_requires_all_pairs(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    store = GateB1EvidenceStore(tmp_path / "evidence", manifest)
    store.append(
        _attempt(
            manifest,
            "GS-T006",
            ExperimentCondition.B0,
            "provider-failure",
            AttemptStatus.INVALID_PROVIDER_ATTEMPT,
        ).model_copy(update={"error_message": "provider 429"})
    )
    result = analyze_gate_b1(manifest, tmp_path / "evidence")
    assert result.invalid_provider_attempts == 1
    assert result.complete_valid_pairs == 0
    assert result.gate_b1_evaluated is False
    assert result.gate_b1_ready_for_review is False


def test_all_ten_pairs_are_ready_for_human_review(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    evidence = tmp_path / "evidence"
    store = GateB1EvidenceStore(evidence, manifest)
    for task_id in GATE_B1_TASK_IDS:
        for condition in GATE_B1_CONDITIONS:
            run_id = f"{condition.value}-{task_id}"
            replacement_of = None
            if task_id == "GS-T006" and condition is ExperimentCondition.B0:
                invalid = _attempt(
                    manifest,
                    task_id,
                    condition,
                    "historical-provider-failure",
                    AttemptStatus.INVALID_PROVIDER_ATTEMPT,
                ).model_copy(update={"error_message": "provider 429"})
                store.append(invalid)
                replacement_of = invalid.run_id
            artifact = _artifact(manifest, task_id, condition, run_id)
            artifact_path = evidence / f"{run_id}.artifact.json"
            artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
            raw_path = evidence / f"{run_id}.raw.json"
            raw_path.write_text(json.dumps({"events": [], "advice_events": [{}]}), encoding="utf-8")
            record = _attempt(
                manifest,
                task_id,
                condition,
                run_id,
                AttemptStatus.VALID_EXPERIMENTAL_RUN,
            ).model_copy(
                update={
                    "artifact_path": str(artifact_path),
                    "raw_evidence_path": str(raw_path),
                    "replacement_of": replacement_of,
                }
            )
            store.append(record, replacement_of=replacement_of)
    result = analyze_gate_b1(manifest, evidence)
    assert result.complete_valid_pairs == 10
    assert result.incomplete_pairs == 0
    assert result.gate_b1_ready_for_review is True
    assert result.gate_b1_evaluated is False


def test_normalized_trajectory_ignores_run_ids_action_ids_and_timestamps() -> None:
    first = [
        {
            "run_id": "run-one",
            "action_id": "action-one",
            "occurred_at": "2026-01-01T00:00:00Z",
            "result": {"tool_name": "run_tests", "success": False, "exit_code": 1},
        }
    ]
    second = [
        {
            "run_id": "run-two",
            "action_id": "action-two",
            "occurred_at": "2027-02-02T00:00:00Z",
            "result": {"tool_name": "run_tests", "success": False, "exit_code": 1},
        }
    ]
    assert normalized_trajectory_signature(first) == normalized_trajectory_signature(second)


def test_normalized_trajectory_detects_tool_and_success_changes() -> None:
    baseline = [{"result": {"tool_name": "read_file", "success": True, "exit_code": 0}}]
    different_tool = [{"result": {"tool_name": "run_tests", "success": True, "exit_code": 0}}]
    different_success = [{"result": {"tool_name": "read_file", "success": False, "exit_code": 1}}]
    assert normalized_trajectory_signature(baseline) != normalized_trajectory_signature(
        different_tool
    )
    assert normalized_trajectory_signature(baseline) != normalized_trajectory_signature(
        different_success
    )
    assert normalized_trajectory_signature(None) is None


@pytest.mark.parametrize(
    (
        "missing_artifact",
        "missing_raw",
        "wrong_o1_review",
        "environment_mismatch",
        "missing_o1_event",
    ),
    [
        (True, False, False, False, False),
        (False, True, False, False, False),
        (False, False, True, False, False),
        (False, False, False, True, False),
        (False, False, False, False, True),
    ],
)
def test_valid_pair_evidence_anomalies_block_readiness(
    tmp_path: Path,
    missing_artifact: bool,
    missing_raw: bool,
    wrong_o1_review: bool,
    environment_mismatch: bool,
    missing_o1_event: bool,
) -> None:
    manifest = _manifest(tmp_path)
    evidence = tmp_path / "evidence"
    _populate_pairs(
        manifest,
        evidence,
        missing_artifact=missing_artifact,
        missing_raw=missing_raw,
        wrong_o1_review=wrong_o1_review,
        environment_mismatch=environment_mismatch,
        missing_o1_event=missing_o1_event,
    )
    result = analyze_gate_b1(manifest, evidence)
    assert result.complete_valid_pairs == 10
    assert result.gate_b1_ready_for_review is False
    assert result.gate_b1_evaluated is False


def test_explicit_execution_mode_runs_injected_path_in_pair_order(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    plan = build_execution_plan(manifest, task_id="GS-T006")
    store = GateB1EvidenceStore(tmp_path / "evidence", manifest)
    called: list[ExperimentCondition] = []

    def callback(item: PlannedGateB1Execution) -> ExperimentExecution:
        called.append(item.condition)
        artifact = _artifact(manifest, item.task_id, item.condition, f"run-{item.condition.value}")
        artifact_path = tmp_path / f"{item.condition.value}.artifact.json"
        raw_path = tmp_path / f"{item.condition.value}.raw.json"
        artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
        raw_path.write_text(json.dumps({"events": [], "advice_events": [{}]}), encoding="utf-8")
        dependencies = AgentDependencies(tmp_path, artifact.run_id, item.task_id)
        return ExperimentExecution(
            artifact,
            artifact_path,
            raw_path,
            dependencies,
            None,
            None,
            tmp_path / f"{item.condition.value}.sqlite",
            manifest.task_map[item.task_id].environment_fingerprint,
        )

    records = execute_gate_b1(
        manifest,
        plan,
        store=store,
        execute_condition=callback,
    )
    assert called == [ExperimentCondition.B0, ExperimentCondition.O1]
    assert all(record.valid_observation for record in records)
    assert all(record.actual_environment_fingerprint for record in records)


def test_runtime_fingerprint_mismatch_is_invalid_infrastructure_evidence(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    plan = build_execution_plan(
        manifest,
        task_id="GS-T006",
        conditions=(ExperimentCondition.B0,),
    )
    store = GateB1EvidenceStore(tmp_path / "evidence", manifest)

    def callback(item: PlannedGateB1Execution) -> ExperimentExecution:
        artifact = _artifact(manifest, item.task_id, item.condition, "mismatched")
        dependencies = AgentDependencies(tmp_path, artifact.run_id, item.task_id)
        return ExperimentExecution(
            artifact,
            tmp_path / "artifact.json",
            tmp_path / "raw.json",
            dependencies,
            None,
            None,
            tmp_path / "steps.sqlite",
            "actual-fingerprint-does-not-match",
        )

    records = execute_gate_b1(
        manifest,
        plan,
        store=store,
        execute_condition=callback,
    )
    assert records[0].status is AttemptStatus.INVALID_INFRASTRUCTURE_ATTEMPT
    assert records[0].valid_observation is False
    assert records[0].actual_environment_fingerprint == "actual-fingerprint-does-not-match"
