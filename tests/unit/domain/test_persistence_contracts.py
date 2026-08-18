from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool

AWARE_TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_minimum_rollout_1_incident_objects_are_representable() -> None:
    task = Task(
        id="GS-T001",
        family_id="GS-F001",
        repository="swesmith/example",
        chronological_index=1,
    )
    run = Run(id="run-001", task_id=task.id, started_at=AWARE_TIMESTAMP)
    tool = Tool(name="run_tests")
    environment = EnvironmentContext(
        id="environment-001",
        repository=task.repository,
        runtime="python-3.12",
    )
    failure = FailureEpisode(
        id="failure-001",
        action_id="action-001",
        failure_type=FailureType.TEST_FAILURE,
        signature="assertion-mismatch",
        symptom="one test failed",
        observed_at=AWARE_TIMESTAMP,
    )
    resolution = Resolution(
        id="resolution-001",
        failure_id=failure.id,
        description="Restore the expected branch condition",
        status=ResolutionStatus.CANDIDATE,
    )
    outcome = Outcome(
        id="outcome-001",
        action_id="action-001",
        success=True,
        tests_passed=10,
        observed_at=AWARE_TIMESTAMP,
    )

    assert run.task_id == task.id
    assert tool.name == "run_tests"
    assert environment.id == "environment-001"
    assert resolution.failure_id == failure.id
    assert outcome.id != outcome.action_id


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (Run, "id"),
        (Task, "id"),
        (Tool, "name"),
        (EnvironmentContext, "id"),
        (FailureEpisode, "id"),
        (Resolution, "id"),
        (Outcome, "id"),
    ],
)
def test_empty_identity_values_are_rejected(model: type[object], field: str) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        if model is Run:
            Run(id="", task_id="task-001", started_at=AWARE_TIMESTAMP)
        elif model is Task:
            Task(
                id="",
                family_id="family-001",
                repository="repo",
                chronological_index=1,
            )
        elif model is Tool:
            Tool(name="")
        elif model is EnvironmentContext:
            EnvironmentContext(id="", repository="repo", runtime="python")
        elif model is FailureEpisode:
            FailureEpisode(
                id="",
                action_id="action-001",
                failure_type=FailureType.TEST_FAILURE,
                signature="signature",
                symptom="symptom",
                observed_at=AWARE_TIMESTAMP,
            )
        elif model is Resolution:
            Resolution(id="", failure_id="failure-001", description="fix")
        else:
            Outcome(id="", action_id="action-001", success=True, observed_at=AWARE_TIMESTAMP)


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: Run(id="run-001", task_id="task-001", started_at=datetime(2026, 1, 1)),
        lambda: FailureEpisode(
            id="failure-001",
            action_id="action-001",
            failure_type=FailureType.TEST_FAILURE,
            signature="signature",
            symptom="symptom",
            observed_at=datetime(2026, 1, 1),
        ),
        lambda: Outcome(
            id="outcome-001",
            action_id="action-001",
            success=True,
            observed_at=datetime(2026, 1, 1),
        ),
    ],
)
def test_naive_timestamps_are_rejected(model_factory: Callable[[], object]) -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        model_factory()


def test_closed_failure_and_resolution_vocabularies_are_validated() -> None:
    with pytest.raises(ValidationError):
        FailureEpisode(
            id="failure-001",
            action_id="action-001",
            failure_type="unknown",  # type: ignore[arg-type]
            signature="signature",
            symptom="symptom",
            observed_at=AWARE_TIMESTAMP,
        )

    with pytest.raises(ValidationError):
        Resolution(
            id="resolution-001",
            failure_id="failure-001",
            description="fix",
            status="unknown",  # type: ignore[arg-type]
        )
