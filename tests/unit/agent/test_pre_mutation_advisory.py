from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from graph_swarm.advisory.formatting import format_advice
from graph_swarm.agent.advisory import prepare_tool_action
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.edit_file import edit_file
from graph_swarm.agent.tools.write_file import write_file
from graph_swarm.domain.advice import (
    AdviceResult,
    ApplicabilityAssessment,
    RecoveryEvidence,
    RecoveryProvenance,
)
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import ResolutionStatus
from graph_swarm.domain.tasks import Task

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def context() -> tuple[Task, EnvironmentContext]:
    return (
        Task(
            id="task-current",
            problem_statement="LassoLexer token highlighting is incorrect.",
            family_id="family-current",
            repository="example/repository",
            chronological_index=20,
        ),
        EnvironmentContext(
            id="environment-current",
            repository="example/repository",
            runtime="python-3.12.1",
        ),
    )


def advice() -> AdviceResult:
    return AdviceResult.historical_recovery(
        matched_failure_episode_id="failure-historical",
        matched_resolution_id="resolution-historical",
        failed_tool="run_command",
        failed_operation="run_command",
        recovery_summary="Restore the token classification before rerunning pytest.",
        resolution_status=ResolutionStatus.OBSERVED_SUCCESSFUL,
            recovery_evidence=RecoveryEvidence(
                successful_observations=1,
                failed_observations=0,
            outcomes=(
                Outcome(
                    id="outcome-historical",
                    action_id="action-successful",
                    success=True,
                    observed_at=NOW,
                ),
            ),
        ),
        applicability=ApplicabilityAssessment(
            matched_fields=("trigger_context",),
            repository="example/repository",
            runtime="python-3.12.1",
        ),
        provenance=RecoveryProvenance(
            failure_episode_id="failure-historical",
            resolution_id="resolution-historical",
            failed_action_id="action-historical",
            environment_id="environment-historical",
            outcome_ids=("outcome-historical",),
        ),
    )


class PrefixService:
    pre_mutation_enabled = True

    def __init__(self, result: AdviceResult) -> None:
        self.result = result
        self.action_calls = 0
        self.prefix_calls: list[dict[str, object]] = []

    def evaluate_action(
        self,
        task: Task,
        action: Any,
        environment: EnvironmentContext,
    ) -> AdviceResult:
        self.action_calls += 1
        return AdviceResult.no_advice("the actual source mutation is not the historical trigger")

    def evaluate_pre_mutation(
        self,
        task: Task,
        action: Any,
        environment: EnvironmentContext,
        prefix_context: dict[str, object],
    ) -> AdviceResult:
        self.prefix_calls.append(prefix_context)
        return self.result

    def render_advice(self, result: AdviceResult) -> str:
        return format_advice(result)


def dependencies(tmp_path: Path, service: PrefixService | None) -> AgentDependencies:
    task, environment = context()
    return AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=environment,
        advisory_service=service,  # pyright: ignore[reportArgumentType]
    )


def test_edit_is_advised_before_source_mutation_and_context_is_prefix_only(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("broken\n", encoding="utf-8")
    service = PrefixService(advice())
    deps = dependencies(tmp_path, service)

    with pytest.raises(ModelRetry):
        prepare_tool_action(
            deps,
            "edit_file",
            "edit_file",
            {
                "path": "module.py",
                "old_text": "broken",
                "new_text": "gold patch expected_pattern",
            },
        )

    assert path.read_text(encoding="utf-8") == "broken\n"
    assert len(service.prefix_calls) == 1
    context_text = repr(service.prefix_calls[0])
    assert "gold patch" not in context_text
    assert "expected_pattern" not in context_text
    assert service.prefix_calls[0]["completed_prefix"] == []
    assert len(deps.advice_events) == 1
    assert deps.advice_events[0].planned_action.tool == "edit_file"


def test_write_is_advised_before_file_creation(tmp_path: Path) -> None:
    service = PrefixService(advice())
    deps = dependencies(tmp_path, service)

    with pytest.raises(ModelRetry):
        prepare_tool_action(
            deps,
            "write_file",
            "write_file",
            {"path": "new.py", "content": "gold patch"},
        )

    assert not (tmp_path / "new.py").exists()
    assert deps.advice_events[0].planned_action.tool == "write_file"


def test_no_eligible_pre_mutation_advice_leaves_edit_execution_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("broken\n", encoding="utf-8")
    service = PrefixService(AdviceResult.no_advice("no eligible pattern"))
    deps = dependencies(tmp_path, service)
    action = prepare_tool_action(
        deps,
        "edit_file",
        "edit_file",
        {"path": "module.py", "old_text": "broken", "new_text": "fixed"},
    )

    result = edit_file(deps, "module.py", "broken", "fixed", action_id=action.id)

    assert result.success
    assert path.read_text(encoding="utf-8") == "fixed\n"
    assert deps.advice_events == []


def test_no_eligible_pre_mutation_advice_leaves_write_execution_unchanged(tmp_path: Path) -> None:
    service = PrefixService(AdviceResult.no_advice("no eligible pattern"))
    deps = dependencies(tmp_path, service)
    action = prepare_tool_action(
        deps,
        "write_file",
        "write_file",
        {"path": "new.py", "content": "fixed"},
    )

    result = write_file(deps, "new.py", "fixed", action_id=action.id)

    assert result.success
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "fixed"
    assert deps.advice_events == []


def test_repeated_equivalent_mutation_does_not_repeat_same_recovery(tmp_path: Path) -> None:
    service = PrefixService(advice())
    deps = dependencies(tmp_path, service)
    arguments: dict[str, object] = {"path": "module.py", "old_text": "a", "new_text": "b"}

    with pytest.raises(ModelRetry):
        prepare_tool_action(deps, "edit_file", "edit_file", arguments)
    second = prepare_tool_action(deps, "edit_file", "edit_file", arguments)

    assert second.tool == "edit_file"
    assert len(deps.advice_events) == 1
    assert len(service.prefix_calls) == 2


def test_completed_prefix_is_available_without_future_or_benchmark_metadata(tmp_path: Path) -> None:
    service = PrefixService(AdviceResult.no_advice("no eligible pattern"))
    deps = dependencies(tmp_path, service)
    prior = prepare_tool_action(
        deps,
        "run_command",
        "run_command",
        {"command": ["python", "-m", "pytest", "tests/test_lasso.py"]},
    )
    from graph_swarm.agent.hooks import emit_action_event
    from graph_swarm.domain.action import ActionResult

    emit_action_event(
        deps,
        ActionResult(
            action_id=prior.id,
            tool_name="run_command",
            success=False,
            exit_code=1,
            output="Name.Builtin token assertion failed",
            started_at=NOW,
            completed_at=NOW,
        ),
    )
    prepare_tool_action(
        deps,
        "write_file",
        "write_file",
        {
            "path": "module.py",
            "content": "FAIL_TO_PASS gold patch expected pattern",
        },
    )

    prefix = service.prefix_calls[-1]
    assert prefix["completed_prefix"]
    assert "FAIL_TO_PASS" not in repr(prefix)
    assert "gold patch" not in repr(prefix)
    assert "expected pattern" not in repr(prefix)
    assert prefix["mutation_boundary"] == {"tool": "write_file", "operation": "write_file"}


def test_without_advisory_service_b0_style_mutation_is_unchanged(tmp_path: Path) -> None:
    deps = dependencies(tmp_path, None)
    action = prepare_tool_action(
        deps,
        "write_file",
        "write_file",
        {"path": "b0.py", "content": "baseline"},
    )

    result = write_file(deps, "b0.py", "baseline", action_id=action.id)

    assert result.success
    assert deps.advice_events == []
