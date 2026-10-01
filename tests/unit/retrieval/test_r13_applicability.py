from datetime import UTC, datetime

from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
    RecoveryTrigger,
)
from graph_swarm.domain.tasks import Task
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
    synthetic = service.evaluate(action("run_command", "run_command"), ENVIRONMENT, pattern)
    mismatched_operation = service.evaluate(action("edit_file", "write_file"), ENVIRONMENT, pattern)

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


def triggered_pattern(*, version_sensitive: bool = True) -> RecoveryPattern:
    pattern = make_pattern(
        applicability_tool="edit_file",
        applicability_operation="edit_file",
    )
    return pattern.model_copy(
        update={
            "environment_constraints": EnvironmentConstraints(
                runtime="python-3.13", versions={"python": "3.13"}
            ),
        }
    ).model_copy(
        update={
            "trigger": RecoveryTrigger(
                failure_type="test_failure",
                failure_signature="run_command:exit_code=1",
                failure_context="DATE_ADD expression referenced before assignment",
                source_task_problem_statement=(
                    "DATE_ADD with three arguments raises UnboundLocalError"
                ),
                source_tool="run_command",
                source_operation="run_command",
                source_action_arguments={
                    "command": ["python", "-m", "pytest", "tests/test_dateadd.py"]
                },
                version_sensitive=version_sensitive,
            )
        }
    )


def trigger_task(*, chronological_index: int = 2) -> Task:
    return Task(
        id="task-current",
        problem_statement="DATE_ADD with three arguments raises an UnboundLocalError.",
        family_id="must-not-be-used",
        repository="example/repository",
        chronological_index=chronological_index,
    )


def trigger_action(command: object) -> PlannedAction:
    result = action("run_command", "run_command")
    return result.model_copy(update={"arguments": {"command": command}})


def test_recovery_action_metadata_does_not_become_failure_trigger() -> None:
    decision = RecoveryPatternApplicabilityService().evaluate(
        action("edit_file", "edit_file"),
        ENVIRONMENT,
        triggered_pattern(),
        trigger_task(),
    )

    assert not decision.applicable
    assert "trigger_tool_mismatch" in decision.rejection_reasons


def test_generic_run_command_does_not_match_every_recovery() -> None:
    decision = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(["git", "status"]),
        ENVIRONMENT,
        triggered_pattern(),
        trigger_task(),
    )

    assert not decision.applicable
    assert "trigger_intent_missing" in decision.rejection_reasons


def test_semantically_relevant_test_trigger_is_eligible() -> None:
    decision = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(["python", "-m", "pytest", "tests/test_dateadd.py"]),
        ENVIRONMENT,
        triggered_pattern(),
        trigger_task(),
    )

    assert decision.applicable
    assert {
        "trigger_tool",
        "trigger_operation",
        "trigger_intent",
        "trigger_context",
    } <= set(decision.matched_fields)


def test_command_alone_does_not_satisfy_trigger_context() -> None:
    command_only = trigger_task().model_copy(
        update={"problem_statement": "command"}
    )
    decision = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(["command"]),
        ENVIRONMENT,
        triggered_pattern(),
        command_only,
    )

    assert not decision.applicable
    assert "trigger_context_mismatch" in decision.rejection_reasons


def test_generic_execution_vocabulary_alone_does_not_satisfy_trigger_context() -> None:
    generic_only = trigger_task().model_copy(
        update={
            "problem_statement": (
                "command run execute execution test testing pytest python "
                "error failure failed exit code"
            )
        }
    )
    decision = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(
            [
                "python",
                "-m",
                "pytest",
                "command",
                "run",
                "execute",
                "error",
                "failure",
                "exit",
                "code",
            ]
        ),
        ENVIRONMENT,
        triggered_pattern(),
        generic_only,
    )

    assert not decision.applicable
    assert "trigger_context_mismatch" in decision.rejection_reasons


def test_one_genuine_shared_semantic_anchor_satisfies_trigger_context() -> None:
    anchored = trigger_task().model_copy(
        update={"problem_statement": "DATE_ADD parsing fails"}
    )
    decision = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(["python", "-m", "pytest", "tests/test_dateadd.py"]),
        ENVIRONMENT,
        triggered_pattern(),
        anchored,
    )

    assert decision.applicable
    assert "trigger_context" in decision.matched_fields


def test_generic_defect_report_adverb_does_not_satisfy_trigger_context() -> None:
    for generic_term in ("incorrectly", "wrong", "wrongly"):
        generic_only = trigger_task().model_copy(
            update={"problem_statement": generic_term}
        )
        decision = RecoveryPatternApplicabilityService().evaluate(
            trigger_action([generic_term]),
            ENVIRONMENT,
            triggered_pattern(),
            generic_only,
        )

        assert not decision.applicable
        assert "trigger_context_mismatch" in decision.rejection_reasons


def test_locale_remains_a_genuine_trigger_context_anchor() -> None:
    locale_task = trigger_task().model_copy(
        update={"problem_statement": "Locale formatting fails"}
    )
    locale_pattern = triggered_pattern().model_copy(
        update={
            "trigger": RecoveryTrigger(
                failure_type="test_failure",
                failure_signature="run_command:exit_code=1",
                failure_context="locale formatting uses the wrong language form",
                source_task_problem_statement="Locale formatting fails",
                source_tool="run_command",
                source_operation="run_command",
                source_action_arguments={"command": ["python", "-m", "pytest"]},
                version_sensitive=True,
            )
        }
    )
    decision = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(["python", "-m", "pytest", "tests/test_locale.py"]),
        ENVIRONMENT,
        locale_pattern,
        locale_task,
    )

    assert decision.applicable
    assert "trigger_context" in decision.matched_fields


def test_run_tests_bridge_preserves_trusted_test_intent() -> None:
    run_tests_trigger = triggered_pattern().model_copy(
        update={
            "trigger": RecoveryTrigger(
                failure_type="test_failure",
                failure_signature="run_tests:exit_code=1",
                failure_context="DATE_ADD expression referenced before assignment",
                source_task_problem_statement=(
                    "DATE_ADD with three arguments raises an UnboundLocalError"
                ),
                source_tool="run_tests",
                source_operation="run_tests",
                source_action_arguments={},
                version_sensitive=True,
            )
        }
    )
    decision = RecoveryPatternApplicabilityService().evaluate(
        action("run_tests", "run_tests"),
        ENVIRONMENT,
        run_tests_trigger,
        trigger_task(),
    )

    assert decision.applicable
    assert "trigger_intent" in decision.matched_fields


def test_unrelated_task_context_is_rejected_without_threshold_tuning() -> None:
    unrelated = trigger_task().model_copy(
        update={"problem_statement": "CSV BOM encoding is detected incorrectly."}
    )
    decision = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(["python", "-m", "pytest", "tests/test_encoding.py"]),
        ENVIRONMENT,
        triggered_pattern(),
        unrelated,
    )

    assert not decision.applicable
    assert "trigger_context_mismatch" in decision.rejection_reasons


def test_generic_variable_overlap_does_not_transfer_recovery_guidance() -> None:
    jinja_scope_task = trigger_task().model_copy(
        update={
            "problem_statement": (
                "Variable aliasing is broken in conditional assignments. "
                "Variables should fall back to an outer scope value."
            )
        }
    )
    decision = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(["python", "-m", "pytest", "tests/test_scope.py"]),
        ENVIRONMENT,
        triggered_pattern(),
        jinja_scope_task,
    )

    assert not decision.applicable
    assert "trigger_context_mismatch" in decision.rejection_reasons


def test_version_sensitive_and_explicitly_version_insensitive_triggers() -> None:
    sensitive = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(["python", "-m", "pytest"]),
        ENVIRONMENT.model_copy(update={"versions": {"python": "3.12"}}),
        triggered_pattern(),
        trigger_task(),
    )
    insensitive = RecoveryPatternApplicabilityService().evaluate(
        trigger_action(["python", "-m", "pytest"]),
        ENVIRONMENT.model_copy(update={"versions": {"python": "3.12"}}),
        triggered_pattern(version_sensitive=False),
        trigger_task(),
    )

    assert not sensitive.applicable
    assert "version_mismatch:python" in sensitive.rejection_reasons
    assert insensitive.applicable
