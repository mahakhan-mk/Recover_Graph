"""Offline tests for the R13b treatment repository view."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.read_models import RecoveryPatternVectorCandidate
from graph_swarm.integration.advisory_runtime import (
    R13B_TREATMENT_PATTERN_IDS,
    NonCanonicalTreatmentPatternError,
    R13bTreatmentRepository,
    TreatmentVectorRepository,
)
from graph_swarm.memory.recovery_embeddings import RecoveryPatternEmbedder
from graph_swarm.retrieval.service import RecoveryPatternRetrievalService


class FakeRepository:
    def __init__(self) -> None:
        self.patterns = tuple(
            _candidate(pattern_id, index + 1)
            for index, pattern_id in enumerate(R13B_TREATMENT_PATTERN_IDS)
        ) + tuple(
            _candidate(pattern_id, 10 + index)
            for index, pattern_id in enumerate(
                (
                    "recovery-pattern-development-a",
                    "recovery-pattern-development-b",
                    "recovery-pattern-development-c",
                )
            )
        )
        self.query_limits: list[int] = []

    def count_recovery_pattern_vectors(self) -> int:
        return len(self.patterns)

    def query_recovery_pattern_vectors(
        self,
        _query_embedding: list[float],
        limit: int,
    ) -> tuple[RecoveryPatternVectorCandidate, ...]:
        self.query_limits.append(limit)
        return self.patterns[:limit]

    def get_recovery_pattern(self, pattern_id: str) -> object:
        return SimpleNamespace(pattern_id=pattern_id)


class FakeEncoder:
    def encode(self, text: str, *, normalize_embeddings: bool) -> object:
        assert text
        assert normalize_embeddings is True
        return [1.0] + [0.0] * 383


def _candidate(pattern_id: str, index: int) -> RecoveryPatternVectorCandidate:
    return RecoveryPatternVectorCandidate(
        pattern=RecoveryPattern(
            id=pattern_id,
            title=f"Pattern {index}",
            guidance="Use the historical resolution.",
            source_failure_id=f"failure-{index}",
            source_resolution_id=f"resolution-{index}",
            source_outcome_id=f"outcome-{index}",
            source_task_id=f"GS-T{index:03d}",
            source_chronological_index=index,
            source_tool="edit_file",
            source_operation="edit_file",
            source_failure_type="test_failure",
            environment_constraints=EnvironmentConstraints(runtime="docker"),
            verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
            evidence_count=1,
            evidence_summary="offline fixture",
            embedding=[0.1] * 384,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        vector_score=1.0 / index,
    )


def test_treatment_count_and_vector_query_are_allowlisted() -> None:
    source = FakeRepository()
    repository = R13bTreatmentRepository(cast(TreatmentVectorRepository, source))

    assert repository.count_recovery_pattern_vectors() == 5
    candidates = repository.query_recovery_pattern_vectors([1.0] + [0.0] * 383, limit=5)

    assert {candidate.pattern.id for candidate in candidates} == R13B_TREATMENT_PATTERN_IDS
    assert source.query_limits == [8, 8]


def test_treatment_refuses_noncanonical_lineage_lookup() -> None:
    source = FakeRepository()
    repository = R13bTreatmentRepository(cast(TreatmentVectorRepository, source))

    with pytest.raises(NonCanonicalTreatmentPatternError):
        repository.get_recovery_pattern("recovery-pattern-development-a")


def test_retrieval_candidates_cannot_contain_historical_development_patterns() -> None:
    source = FakeRepository()
    repository = R13bTreatmentRepository(cast(TreatmentVectorRepository, source))
    retrieval = RecoveryPatternRetrievalService(
        repository,
        embedder=RecoveryPatternEmbedder(encoder=FakeEncoder()),
    )
    task = Task(
        id="IP1",
        problem_statement="Fix the current task.",
        family_id="evaluation-only-family",
        repository="fixture",
        chronological_index=6,
    )
    action = PlannedAction(
        id="action-1",
        run_id="run-1",
        task_id="IP1",
        tool="edit_file",
        operation="edit_file",
        planned_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    environment = EnvironmentContext(
        id="environment-1",
        repository="fixture",
        runtime="docker",
    )

    result = retrieval.retrieve(task, action, environment)

    assert result.selected_pattern is not None
    assert all(
        candidate.pattern_id in R13B_TREATMENT_PATTERN_IDS
        for candidate in result.candidates
    )
