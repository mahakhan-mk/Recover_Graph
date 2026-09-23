from datetime import UTC, datetime

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.retrieval.applicability import RecoveryPatternApplicabilityService

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_pattern(
    *,
    applicability_tool: str | None = None,
    applicability_operation: str | None = None,
) -> RecoveryPattern:
    return RecoveryPattern(
        id="pattern-001",
        title="Observed recovery",
        guidance="Apply the observed correction.",
        source_failure_id="failure-001",
        source_resolution_id="resolution-001",
        source_outcome_id="outcome-001",
        source_task_id="task-001",
        source_chronological_index=1,
        source_tool="run_command",
        source_operation="run_command",
        applicability_tool=applicability_tool,
        applicability_operation=applicability_operation,
        source_failure_type="test_failure",
        environment_constraints=EnvironmentConstraints(runtime="python-3.13"),
        verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
        evidence_count=1,
        evidence_summary="Observed successful recovery.",
        created_at=NOW,
    )


def action(tool: str, operation: str) -> PlannedAction:
    return PlannedAction(
        id="action-001",
        run_id="run-current",
        task_id="task-current",
        tool=tool,
        operation=operation,
        planned_at=NOW,
    )


ENVIRONMENT = EnvironmentContext(
    id="environment-current",
    repository="example/repository",
    runtime="python-3.13",
)


def test_objective_anchored_pattern_uses_real_recovery_action_key() -> None:
    service = RecoveryPatternApplicabilityService()
    pattern = make_pattern(
        applicability_tool="edit_file",
        applicability_operation="edit_file",
    )

    eligible = service.evaluate(action("edit_file", "edit_file"), ENVIRONMENT, pattern)
    synthetic = service.evaluate(
        action("run_command", "run_command"), ENVIRONMENT, pattern
    )
    mismatched_operation = service.evaluate(
        action("edit_file", "write_file"), ENVIRONMENT, pattern
    )

    assert eligible.applicable is True
    assert synthetic.applicable is False
    assert "tool_mismatch" in synthetic.rejection_reasons
    assert mismatched_operation.applicable is False
    assert "operation_mismatch" in mismatched_operation.rejection_reasons


def test_legacy_pattern_falls_back_to_source_key() -> None:
    decision = RecoveryPatternApplicabilityService().evaluate(
        action("run_command", "run_command"),
        ENVIRONMENT,
        make_pattern(),
    )

    assert decision.applicable is True
