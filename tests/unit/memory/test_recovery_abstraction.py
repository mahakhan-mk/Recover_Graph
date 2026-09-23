import json
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from pydantic_ai import PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.recovery_patterns import RecoveryPatternStatus
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.graph.read_models import (
    ActionLineageRecord,
    RecoveryEvidenceLineage,
    RecoveryEvidenceTask,
)
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.memory.recovery_abstraction import (
    FROZEN_RECOVERY_MODEL,
    RECOVERY_MODEL_SETTINGS,
    RecoveryAbstractionModelError,
    RecoveryAbstractionOutput,
    RecoveryAbstractionStructuredOutputError,
    RecoveryAbstractionValidationError,
    RecoveryEvidencePackage,
    abstract_and_persist_recovery_pattern,
    build_recovery_evidence_package,
    construct_recovery_pattern,
    create_recovery_abstraction_agent,
    deterministic_recovery_pattern_id,
    validate_recovery_abstraction,
)
from graph_swarm.research.gate_a1_r8 import recovery_pattern_telemetry
from graph_swarm.settings import Settings

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_settings(*, api_key: str | None = "test-key") -> Settings:
    return Settings(
        neo4j_uri="neo4j://example",
        neo4j_username="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        openrouter_api_key=api_key,
    )


def make_action_record(
    action_id: str,
    tool: str,
    operation: str,
    arguments: dict[str, object],
    *,
    success: bool,
    output: str | None,
    error: str | None,
) -> ActionLineageRecord:
    planned = PlannedAction(
        id=action_id,
        run_id="run-001",
        task_id="task-001",
        tool=tool,
        operation=operation,
        arguments=arguments,
        planned_at=NOW,
    )
    result = ActionResult(
        action_id=action_id,
        tool_name=tool,
        success=success,
        exit_code=0 if success else 1,
        output=output,
        error=error,
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
    )
    return ActionLineageRecord(planned_action=planned, result=result)


def make_lineage(
    *,
    failed_tool: str = "run_tests",
    failed_operation: str = "pytest",
    recovery_tool: str = "write_file",
    recovery_operation: str = "write_file",
    recovery_evidence_source: str | None = None,
    trusted_recovery_action_id: str | None = None,
) -> RecoveryEvidenceLineage:
    failed_action = make_action_record(
        "failed-action-001",
        failed_tool,
        failed_operation,
        {"paths": ["tests"], "options": {"quiet": True}},
        success=False,
        output="1 failed",
        error="tests failed",
    )
    recovery_action = make_action_record(
        "recovery-action-001",
        recovery_tool,
        recovery_operation,
        {"path": "src/example.py", "content": "return value"},
        success=True,
        output="written",
        error=None,
    )
    return RecoveryEvidenceLineage(
        failure=FailureEpisode(
            id="failure-001",
            action_id=failed_action.planned_action.id,
            failure_type=FailureType.TEST_FAILURE,
            signature="pytest:exit_code=1",
            symptom="A test asserted the wrong result.",
            observed_at=NOW,
        ),
        resolution=Resolution(
            id="resolution-001",
            failure_id="failure-001",
            description="Observed concrete change",
            status=ResolutionStatus.OBSERVED_SUCCESSFUL,
            successful_observations=1,
            observed_at=NOW + timedelta(seconds=2),
        ),
        outcome=Outcome(
            id="outcome-001",
            action_id=recovery_action.planned_action.id,
            success=True,
            tests_passed=1,
            tests_failed=0,
            exit_code=0,
            observed_at=NOW + timedelta(seconds=3),
        ),
        task=RecoveryEvidenceTask(
            id="task-001",
            problem_statement="Make the failing test pass.",
            repository="example/repository",
            chronological_index=4,
        ),
        environment=EnvironmentContext(
            id="environment-001",
            repository="example/repository",
            runtime="python-3.13",
            versions={"pytest": "8.0"},
            markers={"platform": "linux"},
        ),
        failed_action=failed_action,
        recovery_action=recovery_action,
        recovery_evidence_source=recovery_evidence_source,
        trusted_recovery_action_id=trusted_recovery_action_id,
    )


def make_model(**updates: object) -> TestModel:
    output: dict[str, object] = {
        "title": "Restore the intended assertion input",
        "guidance": (
            "Inspect the failing test input and make the smallest concrete correction "
            "before rerunning the test."
        ),
        "evidence_summary": (
            "The historical test failure was followed by a concrete file change and "
            "a successful test outcome."
        ),
    }
    output.update(updates)
    return TestModel(custom_output_text=json.dumps(output))


def test_bounded_evidence_package_excludes_benchmark_and_future_metadata() -> None:
    evidence = build_recovery_evidence_package(make_lineage())

    assert "pattern" not in RecoveryEvidenceLineage.model_fields
    assert evidence.source_task_id == "task-001"
    assert evidence.source_chronological_index == 4
    assert evidence.failed_action_tool == "run_tests"
    assert evidence.failed_action_operation == "pytest"
    assert evidence.recovery_action_tool == "write_file"
    assert evidence.recovery_action_operation == "write_file"
    assert evidence.failed_action_arguments["options"] == {"quiet": True}
    assert evidence.source_runtime == "python-3.13"
    serialized = evidence.model_dump_json()
    for forbidden in ("family_id", "mutation", "benchmark", "expected_solution", "future"):
        assert forbidden not in serialized


def test_structured_output_is_typed_and_uses_frozen_model_configuration() -> None:
    repository = Mock(spec=OperationalMemoryRepository)
    pattern = abstract_and_persist_recovery_pattern(
        make_lineage(),
        repository,
        make_settings(),
        model=make_model(),
        created_at=NOW,
    )

    assert isinstance(pattern.title, str)
    assert FROZEN_RECOVERY_MODEL == "cohere/north-mini-code:free"
    assert RECOVERY_MODEL_SETTINGS == {"temperature": 0}
    repository.save_recovery_pattern.assert_called_once_with(pattern)


def test_trusted_provenance_and_environment_populate_pattern() -> None:
    repository = Mock(spec=OperationalMemoryRepository)
    lineage = make_lineage()
    pattern = abstract_and_persist_recovery_pattern(
        lineage,
        repository,
        make_settings(),
        model=make_model(
            title="Use the test failure input as the correction target",
            guidance=(
                "Compare the failing test input with the intended assertion and correct "
                "that input before rerunning tests."
            ),
        ),
        created_at=NOW,
    )

    assert pattern.source_failure_id == lineage.failure.id
    assert pattern.source_resolution_id == lineage.resolution.id
    assert pattern.source_outcome_id == lineage.outcome.id
    assert pattern.source_task_id == lineage.task.id
    assert pattern.source_chronological_index == lineage.task.chronological_index
    assert pattern.source_tool == "run_tests"
    assert pattern.source_operation == "pytest"
    assert pattern.source_failure_type == lineage.failure.failure_type.value
    assert pattern.environment_constraints.runtime == "python-3.13"
    assert pattern.environment_constraints.versions == {"pytest": "8.0"}
    assert pattern.environment_constraints.markers == {"platform": "linux"}
    assert pattern.verification_status is RecoveryPatternStatus.OBSERVED_SUCCESSFUL
    assert pattern.evidence_count == 1
    assert pattern.verification_status is not RecoveryPatternStatus.VERIFIED_FOR_EXPERIMENT


def test_objective_anchored_pattern_separates_failure_and_applicability_keys() -> None:
    repository = Mock(spec=OperationalMemoryRepository)
    lineage = make_lineage(
        failed_tool="run_command",
        failed_operation="run_command",
        recovery_tool="edit_file",
        recovery_operation="edit_file",
        recovery_evidence_source="objective_anchored_v1",
        trusted_recovery_action_id="recovery-action-001",
    )

    pattern = abstract_and_persist_recovery_pattern(
        lineage,
        repository,
        make_settings(),
        model=make_model(),
        created_at=NOW,
    )

    assert pattern.source_tool == "run_command"
    assert pattern.source_operation == "run_command"
    assert pattern.applicability_tool == "edit_file"
    assert pattern.applicability_operation == "edit_file"
    assert pattern.model_dump(mode="json")["applicability_tool"] == "edit_file"

    serialized_r13b_telemetry = json.loads(
        json.dumps(recovery_pattern_telemetry(pattern, "R13b"))
    )
    assert serialized_r13b_telemetry == {
        "pattern_id": pattern.id,
        "verification_status": pattern.verification_status.value,
        "source_tool": "run_command",
        "source_operation": "run_command",
        "applicability_tool": "edit_file",
        "applicability_operation": "edit_file",
    }
    assert recovery_pattern_telemetry(pattern, "R13") == serialized_r13b_telemetry


@pytest.mark.parametrize(
    ("lineage_update", "message"),
    (
        ({"recovery_action": None}, "trusted recovery action"),
        (
            {"trusted_recovery_action_id": "different-action"},
            "one trusted observed mutation action",
        ),
        (
            {
                "recovery_action_candidates": (
                    make_lineage(
                        recovery_evidence_source="objective_anchored_v1",
                        trusted_recovery_action_id="recovery-action-001",
                    ).recovery_action,
                    make_lineage(
                        recovery_evidence_source="objective_anchored_v1",
                        trusted_recovery_action_id="recovery-action-001",
                    ).recovery_action,
                )
            },
            "ambiguous observed-change",
        ),
    ),
)
def test_objective_anchored_lineage_fails_closed(
    lineage_update: dict[str, object],
    message: str,
) -> None:
    lineage = make_lineage(
        recovery_evidence_source="objective_anchored_v1",
        trusted_recovery_action_id="recovery-action-001",
    ).model_copy(update=lineage_update)

    try:
        evidence = build_recovery_evidence_package(lineage)
    except RecoveryAbstractionValidationError as error:
        assert message in str(error)
        return

    with pytest.raises(RecoveryAbstractionValidationError, match=message):
        construct_recovery_pattern(
            RecoveryAbstractionOutput(
                title="Restore the intended assertion input",
                guidance=(
                    "Inspect the failing test input and make a concrete correction "
                    "before rerunning the test."
                ),
                evidence_summary="The concrete change preceded a successful outcome.",
            ),
            evidence,
            source_failure_id=evidence.source_failure_id,
            source_resolution_id=evidence.source_resolution_id,
            source_outcome_id=evidence.source_outcome_id,
            created_at=NOW,
        )


def test_pattern_identity_is_stable_for_identical_source_evidence() -> None:
    first = build_recovery_evidence_package(make_lineage())
    second = build_recovery_evidence_package(make_lineage())

    assert deterministic_recovery_pattern_id(first) == deterministic_recovery_pattern_id(second)


@pytest.mark.parametrize(
    "output",
    (
        RecoveryAbstractionOutput(
            title="Title",
            guidance="",
            evidence_summary="Summary",
        ),
        RecoveryAbstractionOutput(
            title="Title",
            guidance="   ",
            evidence_summary="Summary",
        ),
        RecoveryAbstractionOutput(
            title="Title",
            guidance="Try again.",
            evidence_summary="Summary",
        ),
        RecoveryAbstractionOutput(
            title="family_id lesson",
            guidance="Use the observed action.",
            evidence_summary="Summary",
        ),
        RecoveryAbstractionOutput(
            title="Title",
            guidance="Use the future task transfer annotation.",
            evidence_summary="Summary",
        ),
    ),
)
def test_deterministic_abstraction_validation_rejects_unusable_output(
    output: RecoveryAbstractionOutput,
) -> None:
    with pytest.raises(RecoveryAbstractionValidationError, match="abstraction"):
        validate_recovery_abstraction(output, build_recovery_evidence_package(make_lineage()))


@pytest.mark.parametrize(
    ("guidance", "category"),
    (
        ("Use task-001 when applying this lesson.", "source identifier"),
        ("Use environment-001 when applying this lesson.", "source identifier"),
        ("Use failure-001 when applying this lesson.", "source identifier"),
        ("Use resolution-001 when applying this lesson.", "source identifier"),
        ("Use outcome-001 when applying this lesson.", "source identifier"),
        ("Edit src/example.py before rerunning the check.", "source path"),
        (
            "def compute(value):\n    total = value + 1\n    return total",
            "copied source fragment",
        ),
    ),
)
def test_source_specific_leakage_is_rejected_by_deterministic_category(
    guidance: str,
    category: str,
) -> None:
    output = RecoveryAbstractionOutput(
        title="Operational correction",
        guidance=guidance,
        evidence_summary="The concrete change preceded an objective successful outcome.",
    )
    evidence = build_recovery_evidence_package(make_lineage())
    if category == "copied source fragment":
        evidence = evidence.model_copy(update={"recovery_action_arguments": {"content": guidance}})

    with pytest.raises(RecoveryAbstractionValidationError, match=category):
        validate_recovery_abstraction(output, evidence)


def test_generic_content_phrase_is_not_source_leakage() -> None:
    output = RecoveryAbstractionOutput(
        title="Correct the failing input",
        guidance="Compare the test input with the expected result and return value.",
        evidence_summary="The observed concrete change preceded a successful outcome.",
    )

    assert (
        validate_recovery_abstraction(output, build_recovery_evidence_package(make_lineage()))
        == output
    )


def test_supported_generic_flag_and_version_are_actionable_and_allowed() -> None:
    output = RecoveryAbstractionOutput(
        title="Align the test invocation with the observed correction",
        guidance=(
            "Use the quiet flag with pytest 8.0 when rerunning the focused check, "
            "then compare the result with the expected behavior."
        ),
        evidence_summary=(
            "The supported test configuration preceded an objective successful outcome."
        ),
    )

    assert (
        validate_recovery_abstraction(output, build_recovery_evidence_package(make_lineage()))
        == output
    )


def test_rejected_abstraction_is_not_persisted() -> None:
    repository = Mock(spec=OperationalMemoryRepository)

    with pytest.raises(RecoveryAbstractionValidationError):
        abstract_and_persist_recovery_pattern(
            make_lineage(),
            repository,
            make_settings(),
            model=make_model(guidance="be careful"),
            created_at=NOW,
        )

    repository.save_recovery_pattern.assert_not_called()


def test_model_failure_is_explicit_and_does_not_persist() -> None:
    repository = Mock(spec=OperationalMemoryRepository)

    with pytest.raises(RecoveryAbstractionModelError, match="OpenRouter provider unavailable"):
        abstract_and_persist_recovery_pattern(
            make_lineage(), repository, make_settings(api_key=None), created_at=NOW
        )

    repository.save_recovery_pattern.assert_not_called()


def test_configured_abstraction_model_is_role_specific() -> None:
    settings = make_settings().model_copy(
        update={"openrouter_abstraction_model": "another/model"}
    )
    agent = create_recovery_abstraction_agent(settings)
    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == "another/model"


def test_abstraction_agent_uses_prompted_output() -> None:
    agent = create_recovery_abstraction_agent(make_settings(), model=make_model())
    output_type = cast(PromptedOutput[RecoveryAbstractionOutput], agent.output_type)

    assert isinstance(output_type, PromptedOutput)
    assert output_type.outputs is RecoveryAbstractionOutput


def test_malformed_structured_output_is_explicit_and_does_not_persist() -> None:
    repository = Mock(spec=OperationalMemoryRepository)

    with pytest.raises(RecoveryAbstractionStructuredOutputError):
        abstract_and_persist_recovery_pattern(
            make_lineage(),
            repository,
            make_settings(),
            model=TestModel(custom_output_text=json.dumps({"title": "missing fields"})),
            created_at=NOW,
        )

    repository.save_recovery_pattern.assert_not_called()


def test_model_cannot_supply_provenance_fields() -> None:
    repository = Mock(spec=OperationalMemoryRepository)

    with pytest.raises(RecoveryAbstractionStructuredOutputError):
        abstract_and_persist_recovery_pattern(
            make_lineage(),
            repository,
            make_settings(),
            model=TestModel(
                custom_output_text=json.dumps(
                    {
                        "title": "Restore the intended assertion input",
                        "guidance": (
                            "Inspect the failing test input and make the smallest concrete "
                            "correction before rerunning the test."
                        ),
                        "evidence_summary": (
                            "The historical test failure was followed by a concrete file "
                            "change and a successful test outcome."
                        ),
                        "source_failure_id": "invented-failure",
                    }
                )
            ),
            created_at=NOW,
        )

    repository.save_recovery_pattern.assert_not_called()


def test_repeated_valid_abstraction_persists_stable_identity() -> None:
    repository = Mock(spec=OperationalMemoryRepository)
    lineage = make_lineage()

    first = abstract_and_persist_recovery_pattern(
        lineage, repository, make_settings(), model=make_model(), created_at=NOW
    )
    second = abstract_and_persist_recovery_pattern(
        lineage, repository, make_settings(), model=make_model(), created_at=NOW
    )

    assert first.id == second.id
    assert repository.save_recovery_pattern.call_count == 2
    assert repository.save_recovery_pattern.call_args_list[0].args[0].id == (
        repository.save_recovery_pattern.call_args_list[1].args[0].id
    )


def test_construct_rejects_untrusted_provenance_arguments() -> None:
    evidence = build_recovery_evidence_package(make_lineage())

    with pytest.raises(RecoveryAbstractionValidationError, match="provenance"):
        construct_recovery_pattern(
            RecoveryAbstractionOutput(
                title="Restore the intended assertion input",
                guidance="Inspect the failing test input and make a concrete correction.",
                evidence_summary="The concrete change preceded a successful outcome.",
            ),
            evidence,
            source_failure_id="wrong-failure",
            source_resolution_id=evidence.source_resolution_id,
            source_outcome_id=evidence.source_outcome_id,
            created_at=NOW,
        )


def test_evidence_package_rejects_invalid_source_chronology() -> None:
    with pytest.raises(ValidationError):
        RecoveryEvidencePackage.model_validate(
            build_recovery_evidence_package(make_lineage()).model_dump(mode="python")
            | {"source_chronological_index": -1}
        )
