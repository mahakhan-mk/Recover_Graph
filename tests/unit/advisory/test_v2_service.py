from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import Mock

from graph_swarm.advisory.service import AdvisoryService
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import HistoricalRecoveryAdvice, NoApplicableAdvice
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.read_models import (
    ActionLineageRecord,
    RecoveryPatternLineage,
    RecoveryPatternTask,
)
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.integration.event_persistence import (
    MissingTrustedPlannedActionError,
    persist_agent_event_stream,
)
from graph_swarm.retrieval.service import (
    RecoveryCandidateEvaluation,
    RecoveryPatternRetrievalService,
    RecoveryRetrievalResult,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_context() -> tuple[Task, PlannedAction, EnvironmentContext]:
    task = Task(
        id="task-current",
        problem_statement="Fix the current repository.",
        family_id="family-must-not-enter-retrieval",
        repository="repo-current",
        chronological_index=5,
    )
    action = PlannedAction(
        id="action-current",
        run_id="run-current",
        task_id=task.id,
        tool="write_file",
        operation="write_file",
        arguments={"path": "module.py", "content": "value = 1"},
        planned_at=NOW,
    )
    environment = EnvironmentContext(
        id="environment-current",
        repository=task.repository,
        runtime="python-3.13",
        versions={"pytest": "8.0"},
        markers={"platform": "linux"},
    )
    return task, action, environment


def make_lineage() -> tuple[RecoveryPattern, RecoveryPatternLineage]:
    pattern = RecoveryPattern(
        id="pattern-001",
        title="Correct the implementation",
        guidance="Apply the validated implementation correction.",
        source_failure_id="failure-001",
        source_resolution_id="resolution-001",
        source_outcome_id="outcome-001",
        source_task_id="task-source",
        source_chronological_index=1,
        source_tool="write_file",
        source_operation="write_file",
        source_failure_type="test_failure",
        environment_constraints=EnvironmentConstraints(
            runtime="python-3.13",
            versions={"pytest": "8.0"},
            markers={"platform": "linux"},
        ),
        verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
        evidence_count=1,
        evidence_summary="One objective success.",
        created_at=NOW,
    )
    failed_action = PlannedAction(
        id="action-failed",
        run_id="run-source",
        task_id="task-source",
        tool="write_file",
        operation="write_file",
        arguments={"path": "module.py", "content": "old"},
        planned_at=NOW - timedelta(days=1),
    )
    recovery_action = PlannedAction(
        id="action-recovery",
        run_id="run-source",
        task_id="task-source",
        tool="write_file",
        operation="write_file",
        arguments={"path": "module.py", "content": "new"},
        planned_at=NOW - timedelta(days=1, seconds=-1),
    )
    failure = FailureEpisode(
        id="failure-001",
        action_id=failed_action.id,
        failure_type=FailureType.TEST_FAILURE,
        signature="run_tests:exit_code=1",
        symptom="failed objective",
        observed_at=NOW - timedelta(days=1),
    )
    resolution = Resolution(
        id="resolution-001",
        failure_id=failure.id,
        description="Observed implementation correction.",
        status=ResolutionStatus.OBSERVED_SUCCESSFUL,
        successful_observations=1,
        observed_at=NOW - timedelta(hours=23),
    )
    outcome = Outcome(
        id="outcome-001",
        action_id="action-success",
        success=True,
        exit_code=0,
        observed_at=NOW - timedelta(hours=22),
    )
    environment = EnvironmentContext(
        id="environment-source",
        repository="repo-source",
        runtime="python-3.13",
        versions={"pytest": "8.0"},
        markers={"platform": "linux"},
    )
    lineage = RecoveryPatternLineage(
        pattern=pattern,
        failure=failure,
        resolution=resolution,
        outcome=outcome,
        task=RecoveryPatternTask(
            id="task-source",
            problem_statement="Earlier task.",
            repository="repo-source",
            chronological_index=1,
        ),
        environment=environment,
        failed_action=ActionLineageRecord(
            planned_action=failed_action,
            result=ActionResult(
                action_id=failed_action.id,
                tool_name="write_file",
                success=False,
                started_at=failed_action.planned_at,
                completed_at=failed_action.planned_at,
            ),
        ),
        recovery_action=ActionLineageRecord(
            planned_action=recovery_action,
            result=ActionResult(
                action_id=recovery_action.id,
                tool_name="write_file",
                success=True,
                started_at=recovery_action.planned_at,
                completed_at=recovery_action.planned_at,
            ),
        ),
    )
    return pattern, lineage


class FakeV2Retrieval:
    def __init__(self, result: RecoveryRetrievalResult) -> None:
        self.result = result
        self.calls: list[
            tuple[Task, PlannedAction, EnvironmentContext, Mapping[str, object] | None]
        ] = []

    def retrieve(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        *,
        prefix_context: Mapping[str, object] | None = None,
    ) -> RecoveryRetrievalResult:
        self.calls.append((task, planned_action, environment, prefix_context))
        return self.result


def test_v2_advisory_delegates_and_maps_grounded_lineage_without_repo_equality() -> None:
    task, action, environment = make_context()
    pattern, lineage = make_lineage()
    retrieval = FakeV2Retrieval(
        RecoveryRetrievalResult(
            query_version="v1",
            query_text="safe query",
            candidates=(
                RecoveryCandidateEvaluation(
                    pattern_id=pattern.id,
                    vector_score=0.91,
                    eligible=True,
                    matched_fields=("tool", "operation", "runtime", "versions", "markers"),
                    compatible_versions={"pytest": "8.0"},
                    compatible_markers={"platform": "linux"},
                ),
            ),
            eligible_candidates=(
                RecoveryCandidateEvaluation(
                    pattern_id=pattern.id,
                    vector_score=0.91,
                    eligible=True,
                    matched_fields=("tool", "operation", "runtime", "versions", "markers"),
                    compatible_versions={"pytest": "8.0"},
                    compatible_markers={"platform": "linux"},
                ),
            ),
            selected_pattern=pattern,
            selected_vector_score=0.91,
        )
    )
    repository = Mock(spec=OperationalMemoryRepository)
    repository.get_recovery_pattern.return_value = lineage
    service = AdvisoryService(
        repository,
        retrieval_service=cast(RecoveryPatternRetrievalService, retrieval),
    )

    result = service.evaluate_action(task, action, environment)

    assert result.has_advice
    assert isinstance(result.advice, HistoricalRecoveryAdvice)
    assert result.recovery_summary == pattern.guidance
    assert result.provenance is not None
    assert result.provenance.failure_episode_id == pattern.source_failure_id
    assert result.recovery_evidence is not None
    assert result.recovery_evidence.outcomes[0].id == pattern.source_outcome_id
    assert service.last_retrieval_result is not None
    assert service.last_retrieval_result.selected_pattern is not None
    assert service.last_retrieval_result.selected_pattern.id == pattern.id
    assert service.last_retrieval_result.selected_vector_score == 0.91
    assert retrieval.calls == [(task, action, environment, None)]
    repository.find_historical_recovery_candidates.assert_not_called()

    prefix = {"completed_prefix": [{"tool": "read_file"}]}
    mutation_result = service.evaluate_pre_mutation(
        task,
        action,
        environment,
        prefix,
    )

    assert mutation_result.has_advice
    assert len(retrieval.calls) == 2
    mutation_action = retrieval.calls[1][1]
    assert mutation_action == action
    assert retrieval.calls[1][3] == prefix


def test_v2_advisory_fails_closed_when_source_lineage_is_missing() -> None:
    task, action, environment = make_context()
    pattern, _ = make_lineage()
    retrieval = FakeV2Retrieval(
        RecoveryRetrievalResult(
            query_version="v1",
            query_text="safe query",
            candidates=(
                RecoveryCandidateEvaluation(
                    pattern_id=pattern.id,
                    vector_score=0.91,
                    eligible=True,
                ),
            ),
            eligible_candidates=(
                RecoveryCandidateEvaluation(
                    pattern_id=pattern.id,
                    vector_score=0.91,
                    eligible=True,
                ),
            ),
            selected_pattern=pattern,
            selected_vector_score=0.91,
        )
    )
    repository = Mock(spec=OperationalMemoryRepository)
    repository.get_recovery_pattern.side_effect = KeyError("missing source")

    result = AdvisoryService(
        repository,
        retrieval_service=cast(RecoveryPatternRetrievalService, retrieval),
    ).evaluate_action(task, action, environment)

    assert result.has_advice is False
    assert isinstance(result.advice, NoApplicableAdvice)
    assert "source lineage" in result.advice.reason


def test_t_persistence_fails_closed_without_runtime_planned_action() -> None:
    task, action, environment = make_context()
    event = AgentEvent(
        event_id="event-001",
        run_id=action.run_id,
        task_id=task.id,
        action_id=action.id,
        event_type=AgentEventType.ACTION_COMPLETED,
        result=ActionResult(
            action_id=action.id,
            tool_name=action.tool,
            success=True,
            started_at=NOW,
            completed_at=NOW,
        ),
        occurred_at=NOW,
    )
    repository = Mock(spec=OperationalMemoryRepository)

    try:
        persist_agent_event_stream(
            repository,
            [event],
            task,
            Run(id=action.run_id, task_id=task.id, started_at=NOW),
            environment,
            planned_actions=None,
            require_trusted_planned_actions=True,
        )
    except MissingTrustedPlannedActionError as error:
        assert error.code == "BLOCKED_MISSING_TRUSTED_PLANNED_ACTION_PROVENANCE"
        assert error.action_ids == (action.id,)
    else:
        raise AssertionError("T provenance blocker was not raised")
    repository.save_task.assert_not_called()
