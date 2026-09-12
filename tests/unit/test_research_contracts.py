from datetime import UTC, datetime, timedelta

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import AdviceResult
from graph_swarm.domain.failures import FailureType
from graph_swarm.research.contracts import (
    EXPERIMENT_CONDITIONS,
    ExperimentCondition,
    ExperimentRunArtifact,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_artifact() -> ExperimentRunArtifact:
    planned_action = PlannedAction(
        id="action-001",
        run_id="run-001",
        task_id="task-001",
        tool="run_tests",
        operation="pytest",
        planned_at=NOW,
    )
    return ExperimentRunArtifact(
        experiment_id="GS-E004",
        condition=ExperimentCondition.T,
        run_id="run-001",
        task_id="task-001",
        family_id="GS-F001",
        chronological_index=1,
        model="openai/gpt-oss-120b",
        model_settings={"temperature": 0},
        prompt_version="v1",
        planned_action=planned_action,
        executed_action=ActionResult(
            action_id=planned_action.id,
            tool_name="run_tests",
            success=False,
            exit_code=1,
            started_at=NOW,
            completed_at=NOW + timedelta(seconds=1),
        ),
        advice_received=AdviceResult.no_advice("not applicable"),
        advice_accepted=False,
        failure_type=FailureType.TEST_FAILURE,
        task_success=False,
        known_failure_repeated=True,
        tool_calls=1,
        retries=0,
        input_tokens=100,
        output_tokens=25,
        latency_ms=250.5,
        retrieved_incident_id="failure-001",
        retrieval_score=0.9,
    )


def test_experiment_conditions_have_frozen_ids() -> None:
    assert EXPERIMENT_CONDITIONS == (
        ExperimentCondition.B0,
        ExperimentCondition.O1,
        ExperimentCondition.T,
    )
    assert {condition.name: condition.value for condition in EXPERIMENT_CONDITIONS} == {
        "B0": "B0",
        "O1": "O1",
        "T": "T",
    }


def test_experiment_run_artifact_json_round_trip() -> None:
    artifact = make_artifact()

    restored = ExperimentRunArtifact.model_validate_json(artifact.model_dump_json())

    assert restored == artifact
    assert restored.condition is ExperimentCondition.T
    assert restored.planned_action.tool == "run_tests"
    assert restored.executed_action.exit_code == 1
    assert restored.advice_received is not None
    assert restored.retrieved_incident_id == "failure-001"
