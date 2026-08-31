"""Deterministic, independently testable pre-execution advisory service."""

from datetime import datetime

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.advice import (
    AdviceResult,
    ApplicabilityAssessment,
    RecoveryEvidence,
    RecoveryProvenance,
)
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.resolutions import ResolutionStatus
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.retrieval.applicability import ApplicabilityService
from graph_swarm.retrieval.candidates import HistoricalRecoveryCandidate
from graph_swarm.retrieval.scorer import select_best_candidate


class AdvisoryService:
    """Find and report one applicable historical recovery without intervention."""

    def __init__(
        self,
        repository: OperationalMemoryRepository,
        applicability: ApplicabilityService | None = None,
    ) -> None:
        self._repository = repository
        self._applicability = applicability or ApplicabilityService()

    def evaluate_action(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
    ) -> AdviceResult:
        """Evaluate one planned action against persisted recovery experience.

        Candidate selection is deterministic: advisory-eligible resolution
        status, then descending successful observations, descending successful
        outcome count, descending failure timestamp, and finally ascending
        resolution/failure IDs as stable tie breakers.
        """
        if task.repository != environment.repository:
            return AdviceResult.no_advice(
                "task and current environment repositories do not match"
            )

        candidates = self._repository.find_historical_recovery_candidates(
            planned_action,
            environment,
        )
        eligible = tuple(
            candidate
            for candidate in candidates
            if candidate.resolution.status is ResolutionStatus.OBSERVED_SUCCESSFUL
            and candidate.failed_action.source_run_id != planned_action.run_id
            and _chronologically_available(candidate, planned_action.planned_at)
        )
        if not eligible:
            return AdviceResult.no_advice(
                "no eligible historical recovery candidates matched the planned action",
                considered_candidates=len(candidates),
            )

        applicable = tuple(
            (candidate, self._applicability.evaluate(planned_action, environment, candidate))
            for candidate in eligible
        )
        applicable = tuple(item for item in applicable if item[1].applicable)
        if not applicable:
            return AdviceResult.no_advice(
                "historical recovery candidates were found, but none were applicable",
                considered_candidates=len(candidates),
            )

        candidate = select_best_candidate(tuple(item[0] for item in applicable))
        decision = next(
            decision
            for applicable_candidate, decision in applicable
            if applicable_candidate is candidate
        )
        return _advice_result(
            candidate,
            decision.matched_fields,
            decision.compatible_versions,
            decision.compatible_markers,
        )


def _advice_result(
    candidate: HistoricalRecoveryCandidate,
    matched_fields: tuple[str, ...],
    compatible_versions: dict[str, str],
    compatible_markers: dict[str, str],
) -> AdviceResult:
    return AdviceResult.historical_recovery(
        matched_failure_episode_id=candidate.failure.id,
        matched_resolution_id=candidate.resolution.id,
        failed_tool=candidate.failed_action.tool,
        failed_operation=candidate.failed_action.operation,
        recovery_summary=candidate.resolution.description,
        resolution_status=candidate.resolution.status,
        recovery_evidence=RecoveryEvidence(
            successful_observations=candidate.resolution.successful_observations,
            failed_observations=candidate.resolution.failed_observations,
            outcomes=candidate.outcomes,
        ),
        applicability=ApplicabilityAssessment(
            matched_fields=matched_fields,
            repository=candidate.environment.repository,
            runtime=candidate.environment.runtime,
            compatible_versions=compatible_versions,
            compatible_markers=compatible_markers,
        ),
        provenance=RecoveryProvenance(
            failure_episode_id=candidate.failure.id,
            resolution_id=candidate.resolution.id,
            failed_action_id=candidate.failed_action.id,
            environment_id=candidate.environment.id,
            outcome_ids=tuple(outcome.id for outcome in candidate.outcomes),
        ),
    )


def _chronologically_available(
    candidate: HistoricalRecoveryCandidate,
    planned_at: datetime,
) -> bool:
    """Reject evidence known to occur after the action being evaluated.

    Resolution timestamps were added after some Rollout 1 records existed. For
    legacy rows, the linked outcome timestamp is the conservative recovery
    availability proxy; rows with neither timestamp nor outcome are rejected.
    """
    if candidate.failure.observed_at > planned_at:
        return False
    if candidate.resolution.observed_at is not None:
        if candidate.resolution.observed_at > planned_at:
            return False
    elif not candidate.outcomes:
        return False
    if any(outcome.observed_at > planned_at for outcome in candidate.outcomes):
        return False
    return True
