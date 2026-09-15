from typing import cast

import pytest

from scripts.migrate_track_a_recovery_memory import (
    MigrationBlocked,
    validate_migration_success_invariants,
)


def test_migration_success_invariants_require_complete_regeneration() -> None:
    with pytest.raises(MigrationBlocked, match="source_recovery_lineage_count"):
        validate_migration_success_invariants(
            source_recovery_lineage_count=0,
            regenerated_pattern_count=0,
            regeneration_failure_count=0,
            embedded_pattern_count=0,
            neo4j_validation="NOT_RUN_NO_TRUSTED_LINEAGE",
            vector_index_state="ONLINE",
        )


def test_migration_success_invariants_accept_only_a_complete_pass() -> None:
    validate_migration_success_invariants(
        source_recovery_lineage_count=2,
        regenerated_pattern_count=2,
        regeneration_failure_count=0,
        embedded_pattern_count=2,
        neo4j_validation="PASS",
        vector_index_state="ONLINE",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("regenerated_pattern_count", 1),
        ("regeneration_failure_count", 1),
        ("embedded_pattern_count", 1),
    ),
)
def test_migration_success_invariants_reject_partial_results(field: str, value: int) -> None:
    values = {
        "source_recovery_lineage_count": 2,
        "regenerated_pattern_count": 2,
        "regeneration_failure_count": 0,
        "embedded_pattern_count": 2,
        "neo4j_validation": "PASS",
        "vector_index_state": "ONLINE",
    }
    values[field] = value

    with pytest.raises(MigrationBlocked):
        validate_migration_success_invariants(
            source_recovery_lineage_count=cast(int, values["source_recovery_lineage_count"]),
            regenerated_pattern_count=cast(int, values["regenerated_pattern_count"]),
            regeneration_failure_count=cast(int, values["regeneration_failure_count"]),
            embedded_pattern_count=cast(int, values["embedded_pattern_count"]),
            neo4j_validation=cast(str, values["neo4j_validation"]),
            vector_index_state=cast(str, values["vector_index_state"]),
        )
