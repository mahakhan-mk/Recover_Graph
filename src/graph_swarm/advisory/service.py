"""Deterministic pre-execution advisory service.

The production treatment path uses Recovery Memory V2. The legacy candidate
adapter remains only for older test/reporting doubles that do not implement the
V2 repository surface; it is never selected for a real V2-capable repository.
"""

from collections.abc import Mapping
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
from graph_swarm.graph.read_models import RecoveryPatternLineage
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.retrieval.applicability import ApplicabilityService
from graph_swarm.retrieval.candidates import HistoricalRecoveryCandidate
from graph_swarm.retrieval.scorer import select_best_candidate
from graph_swarm.retrieval.service import (
    RecoveryPatternRetrievalService,
    RecoveryRetrievalResult,
)


class AdvisoryService:
    """Find and report one applicable historical recovery without intervention."""

    # The agent boundary uses this explicit capability flag so legacy service
    # doubles and reporting adapters do not receive prefix-aware lookups.
    pre_mutation_enabled = True

    def __init__(
        self,
        repository: OperationalMemoryRepository,
        applicability: ApplicabilityService | None = None,
        retrieval_service: RecoveryPatternRetrievalService | None = None,
        fail_closed: bool = False,
    ) -> None:
        self._repository = repository
        self._applicability = applicability or ApplicabilityService()
        self._retrieval_service = retrieval_service
        self._fail_closed = fail_closed
        self._v2_repository = (
            repository if _supports_v2(repository) else None
        )
        self.last_retrieval_result: RecoveryRetrievalResult | None = None

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
        if self._retrieval_service is not None or self._v2_repository is not None:
            return self._evaluate_v2(task, planned_action, environment)

        # Compatibility path for legacy reporting/test doubles only. A real
        # Neo4jRepository implements the V2 vector and lineage methods above.

        candidates = self._repository.find_historical_recovery_candidates(
            planned_action,
            environment,
        )
        eligible = tuple(
            candidate
            for candidate in candidates
            if candidate.resolution.status is ResolutionStatus.OBSERVED_SUCCESSFUL
            and any(outcome.success for outcome in candidate.outcomes)
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

    def _evaluate_v2(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        prefix_context: Mapping[str, object] | None = None,
    ) -> AdviceResult:
        retrieval = self._retrieval_service
        if retrieval is None:
            if self._v2_repository is None:
                return AdviceResult.no_advice("V2 retrieval repository is unavailable")
            retrieval = RecoveryPatternRetrievalService(self._v2_repository)
            self._retrieval_service = retrieval

        result = retrieval.retrieve(
            task,
            planned_action,
            environment,
            prefix_context=prefix_context,
        )
        self.last_retrieval_result = result
        if result.selected_pattern is None:
            return AdviceResult.no_advice(
                result.no_selection_reason or "no applicable RecoveryPattern",
                considered_candidates=len(result.candidates),
            )

        selected_evaluation = next(
            (
                candidate
                for candidate in result.eligible_candidates
                if candidate.pattern_id == result.selected_pattern.id
            ),
            None,
        )
        if selected_evaluation is None:
            return AdviceResult.no_advice(
                "selected RecoveryPattern has no eligible audit record",
                considered_candidates=len(result.candidates),
            )

        try:
            lineage = self._repository.get_recovery_pattern(result.selected_pattern.id)
            _validate_pattern_lineage(result.selected_pattern.id, lineage)
        except Exception as error:  # noqa: BLE001 - missing provenance fails closed
            if self._fail_closed:
                raise RuntimeError(
                    "treatment RecoveryPattern source lineage is unavailable"
                ) from error
            return AdviceResult.no_advice(
                "selected RecoveryPattern source lineage is unavailable",
                considered_candidates=len(result.candidates),
            )

        return _v2_advice_result(
            result.selected_pattern.guidance,
            lineage,
            selected_evaluation.matched_fields,
            selected_evaluation.compatible_versions,
            selected_evaluation.compatible_markers,
            environment,
        )

    def evaluate_pre_mutation(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        prefix_context: Mapping[str, object],
    ) -> AdviceResult:
        """Evaluate the same gates at a source-mutation boundary.

        The mutation remains the planned action throughout retrieval and
        applicability. Completed-prefix actions are supplied as evidence for
        a historical trigger when the recovery action and trigger action have
        different tool shapes.
        """
        if self._retrieval_service is None and self._v2_repository is None:
            return AdviceResult.no_advice(
                "pre-mutation lookup requires the V2 retrieval repository"
            )
        return self._evaluate_v2(task, planned_action, environment, prefix_context)


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


def _v2_advice_result(
    guidance: str,
    lineage: RecoveryPatternLineage,
    matched_fields: tuple[str, ...],
    compatible_versions: dict[str, str],
    compatible_markers: dict[str, str],
    environment: EnvironmentContext,
) -> AdviceResult:
    """Map trusted pattern source lineage into the frozen AdviceResult shape."""
    failed_action = lineage.failed_action.planned_action
    return AdviceResult.historical_recovery(
        matched_failure_episode_id=lineage.failure.id,
        matched_resolution_id=lineage.resolution.id,
        failed_tool=failed_action.tool,
        failed_operation=failed_action.operation,
        recovery_summary=guidance,
        resolution_status=lineage.resolution.status,
        recovery_evidence=RecoveryEvidence(
            successful_observations=lineage.resolution.successful_observations,
            failed_observations=lineage.resolution.failed_observations,
            outcomes=(lineage.outcome,),
        ),
        applicability=ApplicabilityAssessment(
            matched_fields=matched_fields,
            repository=environment.repository,
            runtime=environment.runtime,
            compatible_versions=compatible_versions,
            compatible_markers=compatible_markers,
        ),
        provenance=RecoveryProvenance(
            failure_episode_id=lineage.failure.id,
            resolution_id=lineage.resolution.id,
            failed_action_id=failed_action.id,
            environment_id=lineage.environment.id,
            outcome_ids=(lineage.outcome.id,),
        ),
    )


def _validate_pattern_lineage(pattern_id: str, lineage: RecoveryPatternLineage) -> None:
    """Reject a repository response whose source IDs do not match the pattern."""
    pattern = lineage.pattern
    if pattern.id != pattern_id:
        raise ValueError("reconstructed pattern ID does not match the selected pattern")
    if (
        pattern.source_failure_id != lineage.failure.id
        or pattern.source_resolution_id != lineage.resolution.id
        or pattern.source_outcome_id != lineage.outcome.id
        or pattern.source_task_id != lineage.task.id
        or pattern.source_chronological_index != lineage.task.chronological_index
    ):
        raise ValueError("selected RecoveryPattern provenance does not match source lineage")


def _supports_v2(repository: OperationalMemoryRepository) -> bool:
    """Detect the concrete V2 repository surface without invoking it."""
    required = (
        "count_recovery_pattern_vectors",
        "query_recovery_pattern_vectors",
        "get_recovery_pattern",
    )
    return all(callable(getattr(type(repository), name, None)) for name in required)


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
