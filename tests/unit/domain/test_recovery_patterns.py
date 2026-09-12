from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_pattern(**updates: object) -> RecoveryPattern:
    values: dict[str, object] = {
        "id": "pattern-001",
        "title": "Restore the branch condition",
        "guidance": "Inspect the branch polarity and restore the intended condition.",
        "source_failure_id": "failure-001",
        "source_resolution_id": "resolution-001",
        "source_outcome_id": "outcome-001",
        "source_task_id": "task-001",
        "source_chronological_index": 3,
        "source_tool": "run_tests",
        "source_operation": "pytest",
        "source_failure_type": "test_failure",
        "environment_constraints": EnvironmentConstraints(
            runtime="python-3.13",
            versions={"pytest": "8.0"},
            dependencies={"stdlib": "3.13"},
            markers={"platform": "linux"},
        ),
        "verification_status": RecoveryPatternStatus.CANDIDATE,
        "evidence_count": 1,
        "evidence_summary": "One observed change preceded one successful outcome.",
        "created_at": NOW,
    }
    values.update(updates)
    return RecoveryPattern.model_validate(values)


def test_valid_pattern_has_typed_auditable_provenance() -> None:
    pattern = make_pattern()

    assert pattern.id == "pattern-001"
    assert pattern.source_failure_id == "failure-001"
    assert pattern.source_resolution_id == "resolution-001"
    assert pattern.source_outcome_id == "outcome-001"
    assert pattern.source_task_id == "task-001"
    assert pattern.source_chronological_index == 3
    assert pattern.environment_constraints.runtime == "python-3.13"


@pytest.mark.parametrize(
    "field",
    (
        "id",
        "title",
        "guidance",
        "source_failure_id",
        "source_resolution_id",
        "source_outcome_id",
        "source_task_id",
        "source_tool",
        "source_operation",
        "source_failure_type",
        "evidence_summary",
    ),
)
def test_required_text_fields_reject_whitespace(field: str) -> None:
    with pytest.raises(ValidationError):
        make_pattern(**{field: "  "})


def test_guidance_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        make_pattern(guidance=" \t\n ")


def test_verification_lifecycle_serializes_exactly() -> None:
    for status in RecoveryPatternStatus:
        pattern = make_pattern(verification_status=status)
        assert pattern.model_dump(mode="json")["verification_status"] == status.value


def test_counts_and_chronology_reject_negative_values() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        make_pattern(evidence_count=-1)
    with pytest.raises(ValidationError, match="non-negative"):
        make_pattern(source_chronological_index=-1)


def test_timestamps_require_timezone_information() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_pattern(created_at=datetime(2026, 1, 1, 12, 0))
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_pattern(invalidated_at=datetime(2026, 1, 1, 12, 0))


def test_embedding_is_optional_and_round_trips_when_supplied() -> None:
    without_embedding = make_pattern()
    with_embedding = make_pattern(embedding=[0.25, -1.5, 3.0])

    assert without_embedding.embedding is None
    assert with_embedding.model_dump(mode="json")["embedding"] == [0.25, -1.5, 3.0]
    assert RecoveryPattern.model_validate_json(
        with_embedding.model_dump_json()
    ).embedding == [0.25, -1.5, 3.0]


@pytest.mark.parametrize("embedding", ([True, 0.2], [0.1, "0.2"]))
def test_embedding_rejects_booleans_and_non_numeric_values(embedding: object) -> None:
    with pytest.raises(ValidationError, match="embedding"):
        make_pattern(embedding=embedding)


def test_embedding_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="finite"):
        make_pattern(embedding=[0.1, float("nan")])


def test_environment_constraints_are_serializable_without_repository_requirement() -> None:
    pattern = make_pattern()
    serialized = pattern.model_dump(mode="json")

    assert serialized["environment_constraints"] == {
        "runtime": "python-3.13",
        "versions": {"pytest": "8.0"},
        "dependencies": {"stdlib": "3.13"},
        "markers": {"platform": "linux"},
    }
    assert "repository" not in serialized["environment_constraints"]


def test_pattern_rejects_benchmark_and_causal_metadata() -> None:
    with pytest.raises(ValidationError):
        make_pattern(family_id="family-001")

    field_names = set(RecoveryPattern.model_fields)
    assert not any(
        forbidden in field_names
        for forbidden in (
            "family_id",
            "recovery_annotation",
            "mutation_label",
            "gold_patch",
            "expected_solution",
            "future_task_id",
            "caused_by",
        )
    )
