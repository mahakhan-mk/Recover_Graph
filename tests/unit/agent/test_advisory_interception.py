import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import RetryPromptPart
from pydantic_ai.models.test import TestModel

from graph_swarm.advisory.formatting import format_advice
from graph_swarm.advisory.service import AdvisoryService
from graph_swarm.agent.advisory import (
    finalize_pending_advice,
    prepare_tool_action,
    record_post_advice_action,
)
from graph_swarm.agent.coding_agent import create_coding_agent, run_coding_agent
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.run_tests import run_tests
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.behavior import BehaviorChangeEvidence
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.research.artifacts import JsonlResearchArtifactWriter
from graph_swarm.retrieval.candidates import (
    HistoricalActionContext,
    HistoricalRecoveryCandidate,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_context() -> tuple[Task, EnvironmentContext]:
    return (
        Task(
            id="task-current",
            problem_statement="Fix the repository so its tests pass.",
            family_id="family-current",
            repository="example/repository",
            chronological_index=2,
        ),
        EnvironmentContext(
            id="environment-current",
            repository="example/repository",
            runtime="python-3.13",
        ),
    )


def make_candidate() -> HistoricalRecoveryCandidate:
    failure = FailureEpisode(
        id="failure-historical",
        action_id="action-historical",
        failure_type=FailureType.TEST_FAILURE,
        signature="run_tests:exit_code=1",
        symptom="one test failed",
        observed_at=NOW,
    )
    return HistoricalRecoveryCandidate(
        failure=failure,
        failed_action=HistoricalActionContext(
            id=failure.action_id,
            source_run_id="run-historical",
            tool="run_tests",
            operation="run_tests",
            planned_at=NOW,
        ),
        environment=EnvironmentContext(
            id="environment-historical",
            repository="example/repository",
            runtime="python-3.13",
        ),
        resolution=Resolution(
            id="resolution-historical",
            failure_id=failure.id,
            description="Write the corrected implementation before testing again",
            status=ResolutionStatus.OBSERVED_SUCCESSFUL,
            successful_observations=1,
        ),
        outcomes=(
            Outcome(
                id="outcome-historical",
                action_id="action-successful",
                success=True,
                exit_code=0,
                observed_at=NOW,
            ),
        ),
    )


def make_service(candidate: HistoricalRecoveryCandidate) -> AdvisoryService:
    repository = Mock(spec=OperationalMemoryRepository)
    repository.find_historical_recovery_candidates.return_value = (candidate,)
    return AdvisoryService(repository)


def make_settings():
    from graph_swarm.settings import Settings

    return Settings(
        neo4j_uri="neo4j+s://example.databases.neo4j.io",
        neo4j_username="example-user",
        neo4j_password="example-password",
        neo4j_database="example-db",
        agent_request_limit=4,
        agent_tests_timeout_seconds=10,
    )


def test_format_advice_is_concise_deterministic_and_provenanced() -> None:
    task, environment = make_context()
    planned_action = PlannedAction(
        id="planned-action",
        run_id="run-current",
        task_id=task.id,
        tool="run_tests",
        operation="run_tests",
        planned_at=NOW,
    )
    result = make_service(make_candidate()).evaluate_action(
        task,
        planned_action,
        environment,
    )

    rendered = format_advice(result)

    assert rendered == format_advice(result)
    assert "run_tests/run_tests" in rendered
    assert "Write the corrected implementation before testing again" in rendered
    assert "observed_successful" in rendered
    assert "failure_episode_id=failure-historical" in rendered
    assert "not guaranteed" in rendered
    assert len(rendered) < 600


def test_applicable_advice_reaches_pre_tool_flow_before_execution(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    task, environment = make_context()
    artifact_path = tmp_path / "artifacts" / "advice.jsonl"
    dependencies = AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=environment,
        advisory_service=make_service(make_candidate()),
        artifact_writer=JsonlResearchArtifactWriter(artifact_path),
    )
    settings = make_settings()
    model = TestModel(call_tools=["run_tests"], custom_output_text="complete")
    agent = create_coding_agent(settings, model=model)

    result = run_coding_agent(agent, settings, dependencies, "Run the tests.")

    assert result.output == "complete"
    assert len(dependencies.advice_events) == 1
    assert len(dependencies.events) == 1
    assert dependencies.advice_events[0].issued_at < dependencies.events[0].occurred_at
    assert dependencies.behavior_evidence[0].observation == "unchanged"
    assert dependencies.behavior_evidence[0].behavior_changed is False
    assert dependencies.advice_events[0].planned_action.arguments == {}
    assert dependencies.events[0].result.success is True
    retry_prompts = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]
    assert any(
        "A comparable historical action previously failed" in part.content for part in retry_prompts
    )
    artifact_lines = [
        line for line in artifact_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert [json.loads(line)["record_type"] for line in artifact_lines] == [
        "advice_event",
        "behavior_change_evidence",
        "subsequent_outcome",
    ]
    assert "failure-historical" in artifact_lines[0]
    assert "resolution-historical" in artifact_lines[0]
    assert "action_id" in artifact_lines[2]


def test_same_advised_action_retries_once_then_executes_without_a_loop(
    tmp_path: Path,
) -> None:
    task, environment = make_context()
    dependencies = AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=environment,
        advisory_service=make_service(make_candidate()),
    )
    arguments: dict[str, object] = {}

    with pytest.raises(ModelRetry):
        prepare_tool_action(dependencies, "run_tests", "run_tests", arguments)
    repeated_action = prepare_tool_action(
        dependencies,
        "run_tests",
        "run_tests",
        arguments,
    )
    third_action = prepare_tool_action(
        dependencies,
        "run_tests",
        "run_tests",
        arguments,
    )
    (tmp_path / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )
    executed = run_tests(dependencies, timeout_seconds=10, action_id=repeated_action.id)
    record_post_advice_action(dependencies, repeated_action, dependencies.events[-1])

    assert repeated_action.tool == "run_tests"
    assert third_action.tool == "run_tests"
    assert len(dependencies.advice_events) == 1
    assert executed.success is True
    assert len(dependencies.behavior_evidence) == 1
    assert dependencies.behavior_evidence[0].observation == "unchanged"


def test_same_recovery_is_not_repeated_for_a_different_generic_action(
    tmp_path: Path,
) -> None:
    task, environment = make_context()
    dependencies = AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=environment,
        advisory_service=make_service(make_candidate()),
    )

    with pytest.raises(ModelRetry):
        prepare_tool_action(dependencies, "run_tests", "run_tests", {})
    next_action = prepare_tool_action(
        dependencies,
        "run_tests",
        "run_tests",
        {"command": ["git", "status"]},
    )

    assert next_action.tool == "run_tests"
    assert len(dependencies.advice_events) == 1


def test_frozen_one_shot_advice_records_only_the_direct_post_advice_action(
    tmp_path: Path,
) -> None:
    task, environment = make_context()
    service = make_service(make_candidate())
    cast(Any, service).one_shot = True
    dependencies = AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=environment,
        advisory_service=service,
    )

    with pytest.raises(ModelRetry):
        prepare_tool_action(dependencies, "run_tests", "run_tests", {})
    first_after = prepare_tool_action(dependencies, "read_file", "read_file", {"path": "x"})
    record_post_advice_action(dependencies, first_after)
    later = prepare_tool_action(dependencies, "write_file", "write_file", {"path": "x"})
    record_post_advice_action(dependencies, later)

    assert len(dependencies.advice_events) == 1
    assert len(dependencies.behavior_evidence) == 1
    evidence = dependencies.behavior_evidence[0]
    assert evidence.planned_action_before_advice.tool == "run_tests"
    assert evidence.actual_action_after_advice is not None
    assert evidence.actual_action_after_advice.tool == "read_file"


def test_lookup_failure_keeps_normal_tool_execution_available(tmp_path: Path) -> None:
    task, environment = make_context()
    service = Mock(spec=AdvisoryService)
    service.evaluate_action.side_effect = RuntimeError("Neo4j unavailable")
    dependencies = AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=environment,
        advisory_service=service,
    )

    action = prepare_tool_action(dependencies, "read_file", "read_file", {"path": "x"})

    assert action.tool == "read_file"
    assert dependencies.advice_events == []
    assert "Neo4j unavailable" in dependencies.advisory_errors[0]


def test_ignored_advice_is_representable_without_acceptance_inference(tmp_path: Path) -> None:
    task, environment = make_context()
    dependencies = AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=environment,
        advisory_service=make_service(make_candidate()),
    )
    action = PlannedAction(
        id="planned-action",
        run_id=dependencies.run_id,
        task_id=dependencies.task_id,
        tool="run_tests",
        operation="run_tests",
        planned_at=NOW,
    )

    with pytest.raises(ModelRetry):
        prepare_tool_action(dependencies, action.tool, action.operation, action.arguments)
    finalize_pending_advice(dependencies)

    evidence = dependencies.behavior_evidence
    assert len(evidence) == 1
    assert isinstance(evidence[0], BehaviorChangeEvidence)
    assert evidence[0].observation == "no_subsequent_action"
    assert evidence[0].behavior_changed is None
