"""Chronology-safe, deterministic RecoveryPattern retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.recovery_patterns import RecoveryPattern, RecoveryPatternStatus
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.read_models import RecoveryPatternVectorCandidate
from graph_swarm.memory.recovery_embeddings import RecoveryPatternEmbedder
from graph_swarm.retrieval.applicability import RecoveryPatternApplicabilityService
from graph_swarm.retrieval.query import (
    RECOVERY_RETRIEVAL_QUERY_VERSION,
    recovery_retrieval_query_text,
)


class RecoveryPatternVectorRepository(Protocol):
    """Minimal raw-vector repository boundary used by retrieval."""

    def count_recovery_pattern_vectors(self) -> int: ...

    def query_recovery_pattern_vectors(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> tuple[RecoveryPatternVectorCandidate, ...]: ...


class RecoveryCandidateEvaluation(BaseModel):
    """Auditable policy evaluation of one raw vector candidate."""

    model_config = ConfigDict(extra="forbid")

    pattern_id: str
    vector_score: float
    eligible: bool
    rejection_reasons: tuple[str, ...] = ()
    matched_fields: tuple[str, ...] = ()
    compatible_versions: dict[str, str] = Field(default_factory=dict)
    compatible_markers: dict[str, str] = Field(default_factory=dict)

    @field_validator("vector_score", mode="before")
    @classmethod
    def require_finite_score(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("vector_score must be numeric and not boolean")
        if not isfinite(float(value)):
            raise ValueError("vector_score must be finite")
        return value


class RecoveryRetrievalResult(BaseModel):
    """Complete deterministic retrieval decision for later audit logging."""

    model_config = ConfigDict(extra="forbid")

    query_version: str
    query_text: str
    candidates: tuple[RecoveryCandidateEvaluation, ...] = ()
    eligible_candidates: tuple[RecoveryCandidateEvaluation, ...] = ()
    selected_pattern: RecoveryPattern | None = None
    selected_vector_score: float | None = None
    no_selection_reason: str | None = None


class RecoveryPatternRetrievalService:
    """Retrieve only strictly historical, structurally applicable patterns."""

    def __init__(
        self,
        repository: RecoveryPatternVectorRepository,
        embedder: RecoveryPatternEmbedder | None = None,
        applicability: RecoveryPatternApplicabilityService | None = None,
    ) -> None:
        self._repository = repository
        self._embedder = embedder or RecoveryPatternEmbedder()
        self._applicability = applicability or RecoveryPatternApplicabilityService()

    def retrieve(
        self,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        prefix_context: Mapping[str, object] | None = None,
    ) -> RecoveryRetrievalResult:
        """Run query construction, complete-pool search, gates, and ranking."""
        query_text = recovery_retrieval_query_text(task, planned_action, environment)
        pool_size = self._repository.count_recovery_pattern_vectors()
        if pool_size == 0:
            return RecoveryRetrievalResult(
                query_version=RECOVERY_RETRIEVAL_QUERY_VERSION,
                query_text=query_text,
                no_selection_reason="no_vector_candidates",
            )

        query_embedding = self._embedder.embed_text(query_text)
        raw_candidates = self._repository.query_recovery_pattern_vectors(
            query_embedding,
            limit=pool_size,
        )
        patterns_by_id = {candidate.pattern.id: candidate.pattern for candidate in raw_candidates}
        evaluations = tuple(
            sorted(
                (
                    self._evaluate_candidate(
                        candidate,
                        task,
                        planned_action,
                        environment,
                        prefix_context,
                    )
                    for candidate in raw_candidates
                ),
                key=lambda evaluation: (-evaluation.vector_score, evaluation.pattern_id),
            )
        )
        eligible = tuple(evaluation for evaluation in evaluations if evaluation.eligible)
        if not eligible:
            return RecoveryRetrievalResult(
                query_version=RECOVERY_RETRIEVAL_QUERY_VERSION,
                query_text=query_text,
                candidates=evaluations,
                eligible_candidates=eligible,
                no_selection_reason="no_eligible_candidates",
            )

        selected = eligible[0]
        return RecoveryRetrievalResult(
            query_version=RECOVERY_RETRIEVAL_QUERY_VERSION,
            query_text=query_text,
            candidates=evaluations,
            eligible_candidates=eligible,
            selected_pattern=patterns_by_id[selected.pattern_id],
            selected_vector_score=selected.vector_score,
        )

    def _evaluate_candidate(
        self,
        candidate: RecoveryPatternVectorCandidate,
        task: Task,
        planned_action: PlannedAction,
        environment: EnvironmentContext,
        prefix_context: Mapping[str, object] | None,
    ) -> RecoveryCandidateEvaluation:
        pattern = candidate.pattern
        rejection_reasons: list[str] = []
        if pattern.source_chronological_index >= task.chronological_index:
            rejection_reasons.append("not_strictly_historical")

        if pattern.invalidated_at is not None:
            rejection_reasons.append("invalidated")
        elif pattern.verification_status is RecoveryPatternStatus.CANDIDATE:
            rejection_reasons.append("candidate_not_eligible")
        elif pattern.verification_status is RecoveryPatternStatus.STALE:
            rejection_reasons.append("stale")
        elif pattern.verification_status is RecoveryPatternStatus.INVALIDATED:
            rejection_reasons.append("invalidated")

        # No current failure signature exists before this action executes;
        # retrieval therefore never fabricates or compares one.
        applicability = self._applicability.evaluate(
            planned_action,
            environment,
            pattern,
            task,
            prefix_context,
        )
        rejection_reasons.extend(applicability.rejection_reasons)
        return RecoveryCandidateEvaluation(
            pattern_id=pattern.id,
            vector_score=candidate.vector_score,
            eligible=not rejection_reasons,
            rejection_reasons=tuple(rejection_reasons),
            matched_fields=applicability.matched_fields,
            compatible_versions=applicability.compatible_versions,
            compatible_markers=applicability.compatible_markers,
        )


__all__ = [
    "RecoveryCandidateEvaluation",
    "RecoveryPatternRetrievalService",
    "RecoveryPatternVectorRepository",
    "RecoveryRetrievalResult",
]
