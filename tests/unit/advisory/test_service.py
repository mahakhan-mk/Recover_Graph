from datetime import UTC, datetime
from unittest.mock import Mock

from graph_swarm.advisory.service import AdvisoryService
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import AdviceResult
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.retrieval.applicability import ApplicabilityService
from graph_swarm.retrieval.candidates import (
    HistoricalActionContext,
    HistoricalRecoveryCandidate,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_action(*, tool: str = "run_tests", operation: str = "pytest") -> PlannedAction:
    return PlannedAction(
        id="planned-action",
        run_id="run-current",
        task_id="task-current",
        tool=tool,
        operation=operation,
        planned_at=NOW,
    )


def make_environment(
    *,
    repository: str = "example/repository",
    runtime: str = "python-3.13",
    versions: dict[str, str] | None = None,
) -> EnvironmentContext:
    return EnvironmentContext(
        id="environment-current",
        repository=repository,
        runtime=runtime,
        versions=versions or {"pytest": "8.0"},
    )


def make_candidate(
    *,
    failure_id: str = "failure-001",
    resolution_id: str = "resolution-001",
    tool: str = "run_tests",
    operation: str = "pytest",
    repository: str = "example/repository",
    runtime: str = "python-3.13",
    versions: dict[str, str] | None = None,
    status: ResolutionStatus = ResolutionStatus.OBSERVED_SUCCESSFUL,
    successful_observations: int = 1,
    observed_at: datetime = NOW,
) -> HistoricalRecoveryCandidate:
    failure = FailureEpisode(
        id=failure_id,
        action_id=f"{failure_id}-action",
        failure_type=FailureType.TEST_FAILURE,
        signature="run_tests:exit_code=1",
        symptom="one test failed",
        observed_at=observed_at,
    )
    return HistoricalRecoveryCandidate(
        failure=failure,
        failed_action=HistoricalActionContext(
            id=failure.action_id,
            tool=tool,
            operation=operation,
            planned_at=observed_at,
        ),
        environment=EnvironmentContext(
            id=f"{failure_id}-environment",
            repository=repository,
            runtime=runtime,
            versions=versions or {"pytest": "8.0"},
        ),
        resolution=Resolution(
            id=resolution_id,
            failure_id=failure_id,
            description="Restore the expected branch condition",
            status=status,
            successful_observations=successful_observations,
        ),
        outcomes=(
            Outcome(
                id=f"{resolution_id}-outcome",
                action_id=f"{resolution_id}-test-action",
                success=True,
                exit_code=0,
                observed_at=observed_at,
            ),
        ),
    )


def make_service(candidates: tuple[HistoricalRecoveryCandidate, ...]) -> AdvisoryService:
    repository = Mock(spec=OperationalMemoryRepository)
    repository.find_historical_recovery_candidates.return_value = candidates
    return AdvisoryService(repository)


def evaluate(
    service: AdvisoryService,
    *,
    action: PlannedAction | None = None,
    environment: EnvironmentContext | None = None,
) -> AdviceResult:
    return service.evaluate_action(
        Task(
            id="task-current",
            family_id="family-current",
            repository="example/repository",
            chronological_index=1,
        ),
        action or make_action(),
        environment or make_environment(),
    )


def test_no_historical_candidates_returns_explicit_no_advice() -> None:
    result = evaluate(make_service(()))

    assert result.has_advice is False
    assert result.advice.kind == "no_applicable_advice"
    assert result.advice.considered_candidates == 0


def test_observed_successful_match_returns_advice_and_evidence() -> None:
    candidate = make_candidate()

    result = evaluate(make_service((candidate,)))

    assert result.has_advice is True
    assert result.matched_failure_episode_id == candidate.failure.id
    assert result.matched_resolution_id == candidate.resolution.id
    assert result.recovery_summary == candidate.resolution.description
    assert result.resolution_status is ResolutionStatus.OBSERVED_SUCCESSFUL
    assert result.recovery_evidence is not None
    assert result.recovery_evidence.outcomes[0].success is True
    assert result.provenance is not None
    assert result.provenance.failure_episode_id == candidate.failure.id
    assert result.provenance.resolution_id == candidate.resolution.id
    assert result.provenance.failed_action_id == candidate.failed_action.id
    assert result.provenance.environment_id == candidate.environment.id
    assert result.provenance.outcome_ids == (candidate.outcomes[0].id,)


def test_wrong_tool_returns_no_advice() -> None:
    result = evaluate(make_service((make_candidate(tool="write_file"),)))

    assert result.has_advice is False


def test_wrong_operation_returns_no_advice() -> None:
    result = evaluate(make_service((make_candidate(operation="unittest"),)))

    assert result.has_advice is False


def test_incompatible_environment_returns_no_advice() -> None:
    result = evaluate(
        make_service((make_candidate(repository="other/repository"),)),
        environment=make_environment(repository="example/repository"),
    )

    assert result.has_advice is False


def test_incompatible_known_version_returns_no_advice() -> None:
    result = evaluate(
        make_service((make_candidate(versions={"pytest": "7.0"}),)),
        environment=make_environment(versions={"pytest": "8.0"}),
    )

    assert result.has_advice is False


def test_candidate_selection_is_deterministic_and_prefers_stronger_evidence() -> None:
    weaker = make_candidate(
        failure_id="failure-weaker",
        resolution_id="resolution-weaker",
        successful_observations=1,
    )
    stronger = make_candidate(
        failure_id="failure-stronger",
        resolution_id="resolution-stronger",
        successful_observations=2,
    )

    result = evaluate(make_service((weaker, stronger)))

    assert result.matched_resolution_id == stronger.resolution.id


def test_candidate_selection_uses_ids_as_stable_tie_breakers() -> None:
    first = make_candidate(
        failure_id="failure-b",
        resolution_id="resolution-b",
    )
    second = make_candidate(
        failure_id="failure-a",
        resolution_id="resolution-a",
    )

    result = evaluate(make_service((first, second)))

    assert result.matched_resolution_id == "resolution-a"


def test_applicability_service_reports_structured_matching_facts() -> None:
    candidate = make_candidate()
    decision = ApplicabilityService().evaluate(
        make_action(),
        make_environment(),
        candidate,
    )

    assert decision.applicable is True
    assert decision.matched_fields == ("tool", "operation", "repository", "runtime", "versions")
