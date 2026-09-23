"""R13 RecoveryPattern Neo4j read mapping tests."""
# pyright: reportPrivateUsage=false

from datetime import UTC, datetime

from graph_swarm.graph.neo4j_repository import _read_recovery_pattern

NOW = datetime(2026, 1, 1, tzinfo=UTC).isoformat()


def properties(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "pattern-001",
        "title": "Observed recovery",
        "guidance": "Apply the observed correction.",
        "source_failure_id": "failure-001",
        "source_resolution_id": "resolution-001",
        "source_outcome_id": "outcome-001",
        "source_task_id": "task-001",
        "source_chronological_index": 1,
        "source_tool": "run_command",
        "source_operation": "run_command",
        "source_failure_type": "test_failure",
        "verification_status": "observed_successful",
        "evidence_count": 1,
        "evidence_summary": "Observed successful recovery.",
        "created_at": NOW,
    }
    value.update(updates)
    return value


def test_missing_applicability_properties_read_as_none() -> None:
    pattern = _read_recovery_pattern(properties())

    assert pattern.applicability_tool is None
    assert pattern.applicability_operation is None


def test_objective_anchored_applicability_properties_round_trip_from_neo4j() -> None:
    pattern = _read_recovery_pattern(
        properties(
            applicability_tool="edit_file",
            applicability_operation="edit_file",
        )
    )

    assert pattern.source_tool == "run_command"
    assert pattern.source_operation == "run_command"
    assert pattern.applicability_tool == "edit_file"
    assert pattern.applicability_operation == "edit_file"
