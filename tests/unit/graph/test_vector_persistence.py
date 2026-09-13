from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from math import nan
from unittest.mock import Mock, patch

import pytest

from graph_swarm.domain.recovery_patterns import RecoveryPatternStatus
from graph_swarm.graph import queries
from graph_swarm.graph.neo4j_repository import (
    EntityNotFoundError,
    Neo4jRepository,
)


class FakeResult:
    def __init__(self, records: Sequence[Mapping[str, object]]) -> None:
        self.records = records


def make_repository() -> Neo4jRepository:
    with patch("graph_swarm.graph.neo4j_repository.GraphDatabase.driver", return_value=Mock()):
        return Neo4jRepository("uri", "username", "password", "database")


def make_pattern_properties() -> dict[str, object]:
    return {
        "id": "pattern-001",
        "title": "Restore the condition",
        "guidance": "Inspect the branch polarity.",
        "source_failure_id": "failure-001",
        "source_resolution_id": "resolution-001",
        "source_outcome_id": "outcome-001",
        "source_task_id": "task-001",
        "source_chronological_index": 1,
        "source_tool": "run_tests",
        "source_operation": "pytest",
        "source_failure_type": "test_failure",
        "environment_runtime": "python-3.13",
        "environment_versions_json": "{}",
        "environment_dependencies_json": "{}",
        "environment_markers_json": "{}",
        "verification_status": RecoveryPatternStatus.OBSERVED_SUCCESSFUL.value,
        "evidence_count": 1,
        "evidence_summary": "Observed successful recovery.",
        "embedding": [0.1] * 384,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "invalidated_at": None,
    }


def test_embedding_update_validates_and_writes_native_property_only() -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([{"pattern": {}}]),
    ) as execute_query:
        repository.update_recovery_pattern_embedding("pattern-001", [0.5] * 384)

    assert execute_query.call_args.args[0] == queries.UPDATE_RECOVERY_PATTERN_EMBEDDING
    assert execute_query.call_args.kwargs == {
        "pattern_id": "pattern-001",
        "embedding": [0.5] * 384,
    }
    assert "embedding_json" not in execute_query.call_args.args[0]
    assert "SET pattern.embedding = $embedding" in execute_query.call_args.args[0]


def test_embedding_update_requires_existing_pattern() -> None:
    repository = make_repository()

    with patch.object(repository, "execute_query", return_value=FakeResult([])):
        with pytest.raises(EntityNotFoundError, match="was not found"):
            repository.update_recovery_pattern_embedding("pattern-001", [0.5] * 384)


def test_vector_index_creation_is_named_native_and_idempotent() -> None:
    repository = make_repository()

    with patch.object(repository, "execute_query") as execute_query:
        repository.ensure_recovery_pattern_vector_index()
        repository.ensure_recovery_pattern_vector_index()

    assert execute_query.call_count == 2
    assert execute_query.call_args_list[0].args == (
        queries.CREATE_RECOVERY_PATTERN_VECTOR_INDEX,
    )
    assert "IF NOT EXISTS" in queries.CREATE_RECOVERY_PATTERN_VECTOR_INDEX
    assert "RecoveryPattern" in queries.CREATE_RECOVERY_PATTERN_VECTOR_INDEX
    assert "pattern.embedding" in queries.CREATE_RECOVERY_PATTERN_VECTOR_INDEX
    assert "`vector.dimensions`: 384" in queries.CREATE_RECOVERY_PATTERN_VECTOR_INDEX
    assert "`vector.similarity_function`: 'cosine'" in (
        queries.CREATE_RECOVERY_PATTERN_VECTOR_INDEX
    )


def test_vector_query_validates_dimension_finiteness_and_limit_before_database() -> None:
    repository = make_repository()

    with patch.object(repository, "execute_query") as execute_query:
        with pytest.raises(ValueError, match="limit"):
            repository.query_recovery_pattern_vectors([0.1] * 384, 0)
        with pytest.raises(ValueError, match="dimension"):
            repository.query_recovery_pattern_vectors([0.1] * 383, 1)
        with pytest.raises(ValueError, match="finite"):
            repository.query_recovery_pattern_vectors([nan] * 384, 1)

    execute_query.assert_not_called()


def test_vector_query_returns_typed_raw_candidates_and_score() -> None:
    repository = make_repository()
    record = {"pattern": make_pattern_properties(), "vector_score": 0.875}

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([record]),
    ) as execute_query:
        candidates = repository.query_recovery_pattern_vectors([0.1] * 384, 3)

    assert len(candidates) == 1
    assert candidates[0].pattern.id == "pattern-001"
    assert candidates[0].vector_score == 0.875
    assert execute_query.call_args.args[0] == queries.QUERY_RECOVERY_PATTERN_VECTORS
    assert execute_query.call_args.kwargs == {
        "query_embedding": [0.1] * 384,
        "limit": 3,
    }
    assert "recovery_pattern_embedding_idx" in execute_query.call_args.args[0]
    assert "ORDER BY" in execute_query.call_args.args[0]


def test_embedded_pattern_count_is_the_complete_pool_size_primitive() -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([{"count": 7}]),
    ) as execute_query:
        count = repository.count_recovery_pattern_vectors()

    assert count == 7
    execute_query.assert_called_once_with(queries.COUNT_RECOVERY_PATTERN_VECTORS)
    assert "embedding IS NOT NULL" in queries.COUNT_RECOVERY_PATTERN_VECTORS
