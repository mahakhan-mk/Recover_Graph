from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import Agent, AgentCapability, ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from experiments.oracle import FrozenOracleResolver
from graph_swarm.advisory.service import AdvisoryService
from graph_swarm.agent.advisory import prepare_tool_action
from graph_swarm.agent.coding_agent import create_coding_agent
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.tasks import Task
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    BenchmarkTaskCase,
    ExperimentRunArtifactStore,
    ExperimentRunner,
    OracleEvidenceRequired,
    RecurrenceEvaluationRequired,
    load_experiment_configuration,
)
from graph_swarm.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
O1_CONFIG = ROOT / "configs" / "experiments" / "rollout_3_o1.yaml"
PILOT_MANIFEST = ROOT / "benchmark" / "manifests" / "pilot.jsonl"
PILOT_ANNOTATIONS = ROOT / "benchmark" / "annotations" / "recurrence_validation.csv"
PATTERN = "Inspect the failing behavior and restore the validated recovery ordering."


def make_settings() -> Settings:
    return Settings(
        neo4j_uri="neo4j+s://example.databases.neo4j.io",
        neo4j_username="example-user",
        neo4j_password="example-password",
        neo4j_database="example-db",
        groq_api_key="offline-key",
        agent_request_limit=3,
    )


def make_task(task_id: str, index: int) -> Task:
    return Task(
        id=task_id,
        problem_statement=f"Fix only the current problem for {task_id}.",
        family_id="GS-FUTURE",
        repository="offline-repository",
        chronological_index=index,
    )


def make_action(task: Task) -> PlannedAction:
    return PlannedAction(
        id="planned-action",
        run_id="run-current",
        task_id=task.id,
        tool="run_tests",
        operation="run_tests",
        planned_at=datetime.now(UTC),
    )


def test_frozen_oracle_maps_only_transfer_tasks_and_renders_guidance_only() -> None:
    resolver = FrozenOracleResolver.from_frozen_files(
        PILOT_MANIFEST,
        PILOT_ANNOTATIONS,
    )
    transfer = make_task("GS-T006", 6)
    first_occurrence = make_task("GS-T001", 1)
    environment = EnvironmentContext(
        id="environment-current",
        repository=transfer.repository,
        runtime="python",
    )

    advice = resolver.evaluate_action(transfer, make_action(transfer), environment)
    non_transfer_advice = resolver.evaluate_action(
        first_occurrence,
        make_action(first_occurrence),
        environment,
    )

    assert advice.has_advice is True
    assert advice.recovery_summary is not None
    assert non_transfer_advice.has_advice is False
    rendered = resolver.render_advice(advice)
    assert rendered.startswith("Recovery guidance:")
    assert advice.recovery_summary in rendered
    assert "GS-F001" not in rendered
    assert "gold" not in rendered.lower()
    assert "expected_patch" not in rendered
    assert "frozen_transfer_mapping" not in rendered
    assert "oracle-failure" not in rendered
    assert "oracle-resolution" not in rendered


def test_frozen_oracle_ignores_unrelated_filesystem_exploration() -> None:
    resolver = FrozenOracleResolver({"GS-T007": PATTERN})
    task = make_task("GS-T007", 7)
    environment = EnvironmentContext(
        id="environment-current",
        repository=task.repository,
        runtime="python",
    )
    unrelated_action = PlannedAction(
        id="read-action",
        run_id="run-current",
        task_id=task.id,
        tool="read_file",
        operation="read_file",
        arguments={"path": "/workspace/jinja2"},
        planned_at=datetime.now(UTC),
    )

    advice = resolver.evaluate_action(task, unrelated_action, environment)

    assert advice.has_advice is False


def test_b0_receives_no_oracle_advice(tmp_path: Path) -> None:
    task = make_task("GS-T007", 7)
    dependencies = AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=EnvironmentContext(
            id="environment-current",
            repository=task.repository,
            runtime="python",
        ),
    )
    assert prepare_tool_action(
        dependencies,
        "run_tests",
        "run_tests",
        {},
    ).tool == "run_tests"
    assert dependencies.advice_events == []


def test_o1_oracle_delivers_once_before_the_frozen_test_action(tmp_path: Path) -> None:
    resolver = FrozenOracleResolver({"GS-T007": PATTERN})
    task = make_task("GS-T007", 7)
    dependencies = AgentDependencies(
        tmp_path,
        "run-current",
        task.id,
        task=task,
        environment=EnvironmentContext(
            id="environment-current",
            repository=task.repository,
            runtime="python",
        ),
        advisory_service=cast(AdvisoryService, resolver),
    )

    unrelated = PlannedAction(
        id="read-action",
        run_id=dependencies.run_id,
        task_id=task.id,
        tool="read_file",
        operation="read_file",
        arguments={"path": "/workspace/jinja2"},
        planned_at=datetime.now(UTC),
    )
    assert prepare_tool_action(
        dependencies,
        unrelated.tool,
        unrelated.operation,
        unrelated.arguments,
    ).tool == "read_file"
    assert dependencies.advice_events == []

    with pytest.raises(ModelRetry, match=PATTERN):
        prepare_tool_action(dependencies, "run_tests", "run_tests", {})
    assert prepare_tool_action(dependencies, "run_tests", "run_tests", {}).tool == "run_tests"
    assert len(dependencies.advice_events) == 1
    assert dependencies.advice_events[0].planned_action.tool == "run_tests"


def test_o1_injects_guidance_before_action_and_records_o1_evidence(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(O1_CONFIG, project_root=ROOT)
    resolver = FrozenOracleResolver({"GS-T006": PATTERN})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        agent_factory=lambda settings, capabilities: create_coding_agent(
            settings,
            model=TestModel(call_tools=["run_tests"], custom_output_text="complete"),
            capabilities=capabilities,
        ),
        workspace_resolver=lambda _task: workspace,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
        oracle_resolver=resolver,
    )

    execution = runner.run_case(
        BenchmarkTaskCase(make_task("GS-T006", 6), occurrence_index=2)
    )
    assert execution.artifact.condition is ExperimentCondition.O1
    assert execution.artifact.advice_received is not None
    assert execution.artifact.advice_received.recovery_summary == PATTERN
    assert len(execution.dependencies.advice_events) == 1
    assert len(execution.dependencies.behavior_evidence) == 1

    assert execution.agent_result is not None
    retry_text = "\n".join(
        str(part.content)
        for message in execution.agent_result.all_messages()
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    )
    assert PATTERN in retry_text
    assert "GS-F001" not in retry_text
    assert "gold" not in retry_text.lower()
    assert "expected_patch" not in retry_text
    assert "frozen_transfer_mapping" not in retry_text
    assert "oracle-failure" not in retry_text
    assert "oracle-resolution" not in retry_text

    raw = json.loads(execution.raw_evidence_path.read_text(encoding="utf-8"))
    assert raw["condition"] == "O1"
    assert raw["advice_source"] == "O1"
    assert len(raw["advice_events"]) == 1
    assert len(raw["behavior_evidence"]) == 1


def test_non_transfer_o1_has_no_oracle_advice_and_isolated_history(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(O1_CONFIG, project_root=ROOT)
    resolver = FrozenOracleResolver({"GS-T006": PATTERN})
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        prompts.append(str(messages[-1]))
        return ModelResponse(parts=[TextPart("complete")])

    def agent_factory(
        settings: Settings,
        capabilities: Sequence[AgentCapability[AgentDependencies]],
    ) -> Agent[AgentDependencies, str]:
        return create_coding_agent(
            settings,
            model=FunctionModel(respond),
            capabilities=capabilities,
        )

    def workspace_resolver(task: Task) -> Path:
        workspace = tmp_path / "workspaces" / task.id
        workspace.mkdir(parents=True)
        return workspace

    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        agent_factory=agent_factory,
        workspace_resolver=workspace_resolver,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
        oracle_resolver=resolver,
    )

    executions = runner.run_all([make_task("GS-T001", 1), make_task("GS-T002", 2)])

    assert all(not execution.dependencies.advice_events for execution in executions)
    assert all(execution.artifact.advice_received is None for execution in executions)
    assert all(
        not isinstance(execution.dependencies.advisory_service, AdvisoryService)
        for execution in executions
    )
    assert "GS-T002" not in prompts[0]
    assert "GS-T001" not in prompts[1]


def test_missing_o1_transfer_evidence_fails_before_agent_execution(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(O1_CONFIG, project_root=ROOT)
    called = False

    def agent_factory(
        settings: Settings,
        capabilities: Sequence[AgentCapability[AgentDependencies]],
    ) -> Agent[AgentDependencies, str]:
        nonlocal called
        called = True
        return create_coding_agent(settings, capabilities=capabilities)

    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        agent_factory=agent_factory,
        workspace_resolver=lambda task: tmp_path / task.id,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        oracle_resolver=FrozenOracleResolver({}),
    )

    with pytest.raises(OracleEvidenceRequired):
        runner.run_case(
            BenchmarkTaskCase(make_task("GS-T006", 6), occurrence_index=2)
        )
    assert called is False


def test_o1_without_recurrence_matcher_preserves_explicit_outcome_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_experiment_configuration(O1_CONFIG, project_root=ROOT)

    def fake_run(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr("graph_swarm.research.runner.run_coding_agent", fake_run)

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("unused")])

    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        model=FunctionModel(respond),
        workspace_resolver=lambda task: tmp_path / task.id,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        oracle_resolver=FrozenOracleResolver({"GS-T006": PATTERN}),
    )

    with pytest.raises(RecurrenceEvaluationRequired):
        runner.run_case(
            BenchmarkTaskCase(make_task("GS-T006", 6), occurrence_index=2)
        )
