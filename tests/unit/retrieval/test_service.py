from datetime import UTC, datetime
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
from graph_swarm.memory.recovery_embeddings import RecoveryPatternEmbedder
from graph_swarm.retrieval.service import (
    RecoveryPatternRetrievalService,
    RecoveryRetrievalResult,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class FakeEncoder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, text: str, *, normalize_embeddings: bool) -> object:
        assert normalize_embeddings is True
        self.texts.append(text)
        return [0.5] * 384


class FakeRepository:
    def __init__(self, candidates: tuple[RecoveryPatternVectorCandidate, ...]) -> None:
        self.candidates = candidates
        self.count_calls = 0
        self.query_calls: list[tuple[list[float], int]] = []

    def count_recovery_pattern_vectors(self) -> int:
        self.count_calls += 1
        return len(self.candidates)

    def query_recovery_pattern_vectors(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> tuple[RecoveryPatternVectorCandidate, ...]:
        self.query_calls.append((query_embedding, limit))
        return self.candidates


def make_context() -> tuple[Task, PlannedAction, EnvironmentContext]:
    return (
        Task(
            id="current-task",
            problem_statement="Restore the intended branch condition.",
            family_id="must-not-be-read",
            repository="repo-B",
            chronological_index=5,
        ),
        PlannedAction(
            id="current-action",
            run_id="current-run",
            task_id="current-task",
            tool="run_tests",
            operation="pytest",
            planned_at=NOW,
        ),
        EnvironmentContext(
            id="current-environment",
            repository="repo-B",
            runtime="python-3.13",
            versions={"pytest": "8.0"},
            markers={"platform": "linux"},
        ),
    )


def make_pattern(
    pattern_id: str,
    *,
    chronological_index: int = 1,
    status: RecoveryPatternStatus = RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
    runtime: str | None = "python-3.13",
    versions: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    tool: str = "run_tests",
    operation: str = "pytest",
    invalidated_at: datetime | None = None,
) -> RecoveryPattern:
    return RecoveryPattern(
        id=pattern_id,
        title="Restore the branch condition",
        guidance="Inspect the intended condition.",
        source_failure_id=f"{pattern_id}-failure",
        source_resolution_id=f"{pattern_id}-resolution",
        source_outcome_id=f"{pattern_id}-outcome",
        source_task_id=f"{pattern_id}-task",
        source_chronological_index=chronological_index,
        source_tool=tool,
        source_operation=operation,
        source_failure_type="test_failure",
        environment_constraints=EnvironmentConstraints(
            runtime=runtime,
            versions=versions or {"pytest": "8.0"},
            markers=markers or {"platform": "linux"},
            dependencies={"unavailable": "must not be compared"},
        ),
        verification_status=status,
        evidence_count=1,
        evidence_summary="Observed successful recovery.",
        created_at=NOW,
        invalidated_at=invalidated_at,
    )


def make_candidate(
    pattern_id: str,
    score: float,
    **pattern_options: object,
) -> RecoveryPatternVectorCandidate:
    return RecoveryPatternVectorCandidate(
        pattern=make_pattern(
            pattern_id,
            chronological_index=cast(
                int,
                pattern_options.get("chronological_index", 1),
            ),
            status=cast(
                RecoveryPatternStatus,
                pattern_options.get(
                    "status",
                    RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
                ),
            ),
            runtime=cast(str | None, pattern_options.get("runtime", "python-3.13")),
            versions=cast(dict[str, str] | None, pattern_options.get("versions")),
            markers=cast(dict[str, str] | None, pattern_options.get("markers")),
            tool=cast(str, pattern_options.get("tool", "run_tests")),
            operation=cast(str, pattern_options.get("operation", "pytest")),
            invalidated_at=cast(datetime | None, pattern_options.get("invalidated_at")),
        ),
        vector_score=score,
    )


def retrieve(
    candidates: tuple[RecoveryPatternVectorCandidate, ...],
) -> tuple[FakeRepository, RecoveryRetrievalResult]:
    repository = FakeRepository(candidates)
    encoder = FakeEncoder()
    task, action, environment = make_context()
    result = RecoveryPatternRetrievalService(
        repository,
        RecoveryPatternEmbedder(encoder=encoder),
    ).retrieve(task, action, environment)
    return repository, result


def test_retrieval_builds_query_embeds_and_requests_complete_pool() -> None:
    repository, result = retrieve(
        (
            make_candidate("pattern-a", 0.8),
            make_candidate("pattern-b", 0.7),
        )
    )

    assert repository.count_calls == 1
    assert len(repository.query_calls) == 1
    assert repository.query_calls[0][1] == 2
    assert repository.query_calls[0][0] == [0.5] * 384
    assert result.query_version == "v1"
    assert result.selected_pattern is not None


@pytest.mark.parametrize("future_index", (5, 6))
def test_future_high_score_candidate_is_rejected_before_selection(
    future_index: int,
) -> None:
    _, result = retrieve(
        (
            make_candidate("future", 0.99, chronological_index=future_index),
            make_candidate("historical", 0.80, chronological_index=4),
        )
    )

    assert result.selected_pattern is not None
    assert result.selected_pattern.id == "historical"
    assert result.selected_vector_score == 0.80
    future = next(candidate for candidate in result.candidates if candidate.pattern_id == "future")
    assert future.eligible is False
    assert future.rejection_reasons == ("not_strictly_historical",)
    assert all(candidate.pattern_id != "future" for candidate in result.eligible_candidates)


@pytest.mark.parametrize(
    "status",
    (
        RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
        RecoveryPatternStatus.VERIFIED_FOR_EXPERIMENT,
    ),
)
def test_development_lifecycle_allows_observed_and_verified_patterns(
    status: RecoveryPatternStatus,
) -> None:
    _, result = retrieve((make_candidate("pattern", 0.9, status=status),))

    assert result.selected_pattern is not None
    assert result.selected_pattern.id == "pattern"
    assert result.candidates[0].rejection_reasons == ()


@pytest.mark.parametrize(
    ("status", "invalidated_at", "reason"),
    (
        (RecoveryPatternStatus.CANDIDATE, None, "candidate_not_eligible"),
        (RecoveryPatternStatus.STALE, None, "stale"),
        (RecoveryPatternStatus.INVALIDATED, None, "invalidated"),
        (RecoveryPatternStatus.OBSERVED_SUCCESSFUL, NOW, "invalidated"),
    ),
)
def test_lifecycle_policy_rejects_ineligible_patterns(
    status: RecoveryPatternStatus,
    invalidated_at: datetime | None,
    reason: str,
) -> None:
    _, result = retrieve(
        (make_candidate("pattern", 0.9, status=status, invalidated_at=invalidated_at),)
    )

    assert result.selected_pattern is None
    assert result.no_selection_reason == "no_eligible_candidates"
    assert result.candidates[0].rejection_reasons == (reason,)


@pytest.mark.parametrize(
    ("pattern_options", "reason"),
    (
        ({"tool": "write_file"}, "tool_mismatch"),
        ({"operation": "unittest"}, "operation_mismatch"),
        ({"runtime": "python-3.12"}, "runtime_mismatch"),
        ({"versions": {"pytest": "7.0"}}, "version_mismatch:pytest"),
        ({"markers": {"platform": "windows"}}, "marker_mismatch:platform"),
    ),
)
def test_structural_and_environment_mismatches_are_explicit(
    pattern_options: dict[str, object],
    reason: str,
) -> None:
    _, result = retrieve((make_candidate("pattern", 0.9, **pattern_options),))

    assert result.selected_pattern is None
    assert reason in result.candidates[0].rejection_reasons


def test_cross_repository_transfer_and_missing_dependency_facts_are_allowed() -> None:
    _, result = retrieve(
        (
            make_candidate(
                "pattern-from-repo-A",
                0.9,
                versions={"unavailable": "1.0"},
                markers={},
            ),
        )
    )

    assert result.selected_pattern is not None
    assert result.selected_pattern.id == "pattern-from-repo-A"
    assert not any("repository" in reason for reason in result.candidates[0].rejection_reasons)


def test_eligible_candidates_rank_by_score_then_pattern_id_without_fusion() -> None:
    _, result = retrieve(
        (
            make_candidate("pattern-b", 0.8),
            make_candidate("pattern-a", 0.8),
            make_candidate("pattern-c", 0.7),
        )
    )

    assert [candidate.pattern_id for candidate in result.eligible_candidates] == [
        "pattern-a",
        "pattern-b",
        "pattern-c",
    ]
    assert result.selected_pattern is not None
    assert result.selected_pattern.id == "pattern-a"
    assert result.selected_vector_score == 0.8
    assert [candidate.vector_score for candidate in result.candidates] == [0.8, 0.8, 0.7]


def test_no_vector_candidates_returns_explicit_no_selection() -> None:
    repository, result = retrieve(())

    assert repository.query_calls == []
    assert result.selected_pattern is None
    assert result.no_selection_reason == "no_vector_candidates"


def test_no_eligible_candidates_preserves_all_rejection_reasons() -> None:
    _, result = retrieve(
        (
            make_candidate(
                "rejected",
                0.9,
                chronological_index=5,
                tool="write_file",
                status=RecoveryPatternStatus.STALE,
            ),
        )
    )

    assert result.selected_pattern is None
    assert result.no_selection_reason == "no_eligible_candidates"
    assert result.candidates[0].rejection_reasons == (
        "not_strictly_historical",
        "stale",
        "tool_mismatch",
    )
    assert "family_id" not in result.model_dump()
