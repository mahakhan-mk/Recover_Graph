"""Opt-in live validation for the frozen RecoveryPattern abstraction model."""

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.graph.read_models import (
    ActionLineageRecord,
    RecoveryEvidenceLineage,
    RecoveryEvidenceTask,
)
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.memory.recovery_abstraction import (
    FROZEN_RECOVERY_MODEL,
    RecoveryAbstractionOutput,
    abstract_and_persist_recovery_pattern,
    build_recovery_evidence_package,
    deterministic_recovery_pattern_id,
    validate_recovery_abstraction,
)
from graph_swarm.settings import get_settings

RUN_OPENROUTER_INTEGRATION = os.getenv("GRAPH_SWARM_RUN_OPENROUTER_INTEGRATION") == "1"
NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_live_evidence() -> RecoveryEvidenceLineage:
    def action_record(
        action_id: str,
        tool: str,
        operation: str,
        arguments: dict[str, object],
        success: bool,
    ) -> ActionLineageRecord:
        planned = PlannedAction(
            id=action_id,
            run_id="run-live-001",
            task_id="task-live-001",
            tool=tool,
            operation=operation,
            arguments=arguments,
            planned_at=NOW,
        )
        return ActionLineageRecord(
            planned_action=planned,
            result=ActionResult(
                action_id=action_id,
                tool_name=tool,
                success=success,
                exit_code=0 if success else 1,
                output="passed" if success else "failed",
                error=None if success else "test failure",
                started_at=NOW,
                completed_at=NOW + timedelta(seconds=1),
            ),
        )

    failed_action = action_record(
        "failed-action-live-001",
        "run_tests",
        "pytest",
        {"paths": ["tests"], "options": {"quiet": True}},
        False,
    )
    recovery_action = action_record(
        "recovery-action-live-001",
        "write_file",
        "write_file",
        {"path": "src/example.py", "content": "return value"},
        True,
    )
    return RecoveryEvidenceLineage(
        failure=FailureEpisode(
            id="failure-live-001",
            action_id=failed_action.planned_action.id,
            failure_type=FailureType.TEST_FAILURE,
            signature="pytest:exit_code=1",
            symptom="The assertion input did not match the intended result.",
            observed_at=NOW,
        ),
        resolution=Resolution(
            id="resolution-live-001",
            failure_id="failure-live-001",
            description="Observed concrete file change",
            status=ResolutionStatus.OBSERVED_SUCCESSFUL,
            successful_observations=1,
            observed_at=NOW + timedelta(seconds=2),
        ),
        outcome=Outcome(
            id="outcome-live-001",
            action_id=recovery_action.planned_action.id,
            success=True,
            tests_passed=1,
            tests_failed=0,
            exit_code=0,
            observed_at=NOW + timedelta(seconds=3),
        ),
        task=RecoveryEvidenceTask(
            id="task-live-001",
            problem_statement="Make the failing assertion pass.",
            repository="example/repository",
            chronological_index=1,
        ),
        environment=EnvironmentContext(
            id="environment-live-001",
            repository="example/repository",
            runtime="python-3.13",
            versions={"pytest": "8.0"},
            markers={"platform": "linux"},
        ),
        failed_action=failed_action,
        recovery_action=recovery_action,
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_OPENROUTER_INTEGRATION,
    reason="Set GRAPH_SWARM_RUN_OPENROUTER_INTEGRATION=1 to run against OpenRouter",
)
def test_frozen_openrouter_recovery_abstraction_returns_typed_output() -> None:
    repository = Mock(spec=OperationalMemoryRepository)
    evidence = make_live_evidence()
    pattern = abstract_and_persist_recovery_pattern(
        evidence,
        repository,
        get_settings(),
        created_at=NOW,
    )

    assert FROZEN_RECOVERY_MODEL == "cohere/north-mini-code:free"
    assert pattern.id == deterministic_recovery_pattern_id(
        build_recovery_evidence_package(evidence)
    )
    assert pattern.title.strip()
    assert pattern.guidance.strip()
    assert pattern.evidence_summary.strip()
    assert pattern.source_failure_id == "failure-live-001"
    assert pattern.source_resolution_id == "resolution-live-001"
    assert pattern.source_outcome_id == "outcome-live-001"
    assert pattern.source_task_id == "task-live-001"
    assert pattern.verification_status.value == "observed_successful"
    assert pattern.evidence_count == 1
    assert pattern.embedding is None
    validate_recovery_abstraction(
        RecoveryAbstractionOutput(
            title=pattern.title,
            guidance=pattern.guidance,
            evidence_summary=pattern.evidence_summary,
        ),
        build_recovery_evidence_package(evidence),
    )
    repository.save_recovery_pattern.assert_called_once_with(pattern)
