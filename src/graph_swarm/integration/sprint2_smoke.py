"""Read-only Integration Point 1 checks for the real R13b advisory path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from graph_swarm.domain.recovery_patterns import RecoveryPatternStatus
from graph_swarm.graph.read_models import RecoveryPatternLineage, RecoveryPatternVectorCandidate
from graph_swarm.memory.recovery_embeddings import RECOVERY_PATTERN_EMBEDDING_DIMENSION
from graph_swarm.retrieval.service import RecoveryRetrievalResult

R13B_CANONICAL_PATTERN_IDS: dict[str, str] = {
    "GS-T001": "recovery-pattern-ad07a6a45718b848a30ad377",
    "GS-T002": "recovery-pattern-162e3999c4a2c66a1ff647ed",
    "GS-T003": "recovery-pattern-f490f62ab931191c6eac6db1",
    "GS-T004": "recovery-pattern-99a54266f940e1d4648f4698",
    "GS-T005": "recovery-pattern-fd7b65022b22dc5f2a42816f",
}
VECTOR_INDEX_NAME = "recovery_pattern_embedding_idx"


class ReadOnlyR13bRepository(Protocol):
    """Repository surface used by preflight; all methods are read-only."""

    def count_recovery_pattern_vectors(self) -> int:
        ...

    def query_recovery_pattern_vectors(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> tuple[RecoveryPatternVectorCandidate, ...]:
        ...

    def get_recovery_pattern(self, pattern_id: str) -> RecoveryPatternLineage:
        ...


@dataclass(frozen=True)
class CorpusPatternCheck:
    """Evidence for one canonical pattern in the read-only preflight."""

    task_id: str
    pattern_id: str
    verification_status: str
    source_chronological_index: int
    embedding_dimension: int
    failure_episode_id: str
    resolution_id: str
    outcome_id: str
    source_task_id: str


@dataclass(frozen=True)
class CorpusPreflight:
    """Read-only R13b corpus/index observations."""

    vector_index: str
    embedded_pattern_count: int
    queryable_candidate_count: int
    canonical_patterns: tuple[CorpusPatternCheck, ...]


def preflight_r13b_corpus(repository: ReadOnlyR13bRepository) -> CorpusPreflight:
    """Verify canonical patterns, provenance, dimensions, and vector querying.

    The vector query uses a deterministic valid 384-dimensional probe only to
    prove that the existing native index is queryable and to enumerate the
    embedded corpus.  It is not used as an experiment retrieval decision.
    """
    count = repository.count_recovery_pattern_vectors()
    if count < len(R13B_CANONICAL_PATTERN_IDS):
        raise ValueError(
            f"R13b embedded pattern count {count} is below the canonical corpus size"
        )
    probe = [1.0] + [0.0] * (RECOVERY_PATTERN_EMBEDDING_DIMENSION - 1)
    candidates = repository.query_recovery_pattern_vectors(probe, limit=count)
    by_id = {candidate.pattern.id: candidate for candidate in candidates}
    checks: list[CorpusPatternCheck] = []
    for task_id, pattern_id in R13B_CANONICAL_PATTERN_IDS.items():
        candidate = by_id.get(pattern_id)
        if candidate is None:
            raise ValueError(f"canonical pattern {pattern_id!r} was not returned by the index")
        lineage = repository.get_recovery_pattern(pattern_id)
        _validate_canonical_pattern(task_id, candidate, lineage)
        checks.append(
            CorpusPatternCheck(
                task_id=task_id,
                pattern_id=pattern_id,
                verification_status=candidate.pattern.verification_status.value,
                source_chronological_index=candidate.pattern.source_chronological_index,
                embedding_dimension=len(candidate.pattern.embedding or []),
                failure_episode_id=lineage.failure.id,
                resolution_id=lineage.resolution.id,
                outcome_id=lineage.outcome.id,
                source_task_id=lineage.task.id,
            )
        )
    return CorpusPreflight(
        vector_index=VECTOR_INDEX_NAME,
        embedded_pattern_count=count,
        queryable_candidate_count=len(candidates),
        canonical_patterns=tuple(checks),
    )


def _validate_canonical_pattern(
    task_id: str,
    candidate: RecoveryPatternVectorCandidate,
    lineage: RecoveryPatternLineage,
) -> None:
    pattern = candidate.pattern
    if pattern.verification_status is not RecoveryPatternStatus.OBSERVED_SUCCESSFUL:
        raise ValueError(f"{task_id} pattern is not observed_successful")
    if len(pattern.embedding or []) != RECOVERY_PATTERN_EMBEDDING_DIMENSION:
        raise ValueError(f"{task_id} pattern embedding dimension is not 384")
    if pattern.source_chronological_index != int(task_id.removeprefix("GS-T")):
        raise ValueError(f"{task_id} pattern chronology is inconsistent")
    if lineage.task.id != task_id:
        raise ValueError(f"{task_id} pattern source task is {lineage.task.id!r}")
    if not lineage.outcome.success:
        raise ValueError(f"{task_id} pattern source outcome was not successful")
    if lineage.resolution.status.value != RecoveryPatternStatus.OBSERVED_SUCCESSFUL.value:
        raise ValueError(f"{task_id} pattern source resolution was not observed_successful")
    if (
        pattern.source_failure_id != lineage.failure.id
        or pattern.source_resolution_id != lineage.resolution.id
        or pattern.source_outcome_id != lineage.outcome.id
        or pattern.source_task_id != lineage.task.id
        or pattern.source_tool != lineage.failed_action.planned_action.tool
        or pattern.source_operation != lineage.failed_action.planned_action.operation
    ):
        raise ValueError(f"{task_id} pattern provenance is incomplete or inconsistent")


def chronology_audit(
    result: RecoveryRetrievalResult,
    *,
    task_chronological_index: int,
    pattern_lineages: dict[str, RecoveryPatternLineage],
) -> dict[str, tuple[int, ...]]:
    """Return selected/rejected source indexes to make chronology auditable."""
    selected = () if result.selected_pattern is None else (
        pattern_lineages[result.selected_pattern.id].task.chronological_index,
    )
    rejected = tuple(
        sorted(
            pattern_lineages[item.pattern_id].task.chronological_index
            for item in result.candidates
            if not item.eligible
            and "not_strictly_historical" in item.rejection_reasons
        )
    )
    if any(index >= task_chronological_index for index in selected):
        raise ValueError("retrieval selected a non-historical pattern")
    if any(index < task_chronological_index for index in rejected):
        raise ValueError("retrieval rejected an earlier pattern for chronology")
    return {"selected": selected, "not_strictly_historical": rejected}


__all__ = [
    "CorpusPatternCheck",
    "CorpusPreflight",
    "R13B_CANONICAL_PATTERN_IDS",
    "VECTOR_INDEX_NAME",
    "chronology_audit",
    "preflight_r13b_corpus",
]
