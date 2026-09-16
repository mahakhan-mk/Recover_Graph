from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic_ai import Agent, AgentCapability, AgentRunResult
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from graph_swarm.agent.coding_agent import create_coding_agent
from graph_swarm.agent.dependencies import AgentDependencies, ExecutionRuntime
from graph_swarm.detection.failure_detector import detect_failure
from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.events import AgentEvent, AgentEventType
from graph_swarm.domain.failures import FailureType
from graph_swarm.domain.tasks import Task
from graph_swarm.research.contracts import ExperimentCondition
from graph_swarm.research.runner import (
    BaselineWorkspaceManager,
    BenchmarkEvaluation,
    BenchmarkTaskCase,
    ExperimentConfiguration,
    ExperimentLimits,
    ExperimentRunArtifactStore,
    ExperimentRunner,
    RecurrenceEvaluationRequired,
    benchmark_recurrence_determination,
    build_task_prompt,
    load_experiment_configuration,
    load_tasks,
)
from graph_swarm.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
PILOT_CONFIG = ROOT / "configs" / "experiments" / "rollout_3_pilot.yaml"
O1_CONFIG = ROOT / "configs" / "experiments" / "rollout_3_o1.yaml"
ACTIVE_B0_CONFIG = ROOT / "configs" / "experiments" / "rollout_3a_pilot.yaml"
ACTIVE_O1_CONFIG = ROOT / "configs" / "experiments" / "rollout_3a_o1.yaml"


def make_settings() -> Settings:
    return Settings(
        neo4j_uri="neo4j+s://example.databases.neo4j.io",
        neo4j_username="example-user",
        neo4j_password="example-password",
        neo4j_database="example-db",
        groq_api_key="offline-key",
        agent_request_limit=3,
    )


def make_task(task_id: str, index: int, family_id: str = "GS-FUTURE") -> Task:
    return Task(
        id=task_id,
        problem_statement=f"Fix only the current problem for {task_id}.",
        family_id=family_id,
        repository="offline-repository",
        chronological_index=index,
    )


def test_gs_e003_pilot_configuration_is_b0_only() -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)

    assert configuration.config.experiment_id == "GS-E003"
    assert configuration.config.development is True
    assert configuration.config.pilot is True
    assert configuration.config.conditions == (ExperimentCondition.B0,)
    assert configuration.config.limits.max_actions == 20
    assert configuration.config.limits.max_requests == 24
    assert configuration.config.limits.timeout_seconds == 300
    assert configuration.config.model_config_path == "configs/models/groq.yaml"
    assert configuration.model.model == "openai/gpt-oss-120b"
    assert configuration.model.prompt_version == "v1"


def test_experiment_limits_request_budget_is_optional_and_positive() -> None:
    assert ExperimentLimits(max_actions=20, timeout_seconds=300).max_requests is None
    assert (
        ExperimentLimits(
            max_actions=20,
            max_requests=24,
            timeout_seconds=300,
        ).max_requests
        == 24
    )


def test_existing_experiment_configuration_without_request_budget_remains_valid() -> None:
    configuration = ExperimentConfiguration.model_validate(
        {
            "experiment_id": "legacy",
            "rollout": "legacy_rollout",
            "model_config": "configs/models/groq.yaml",
            "conditions": ["B0"],
            "limits": {"max_actions": 20, "timeout_seconds": 300},
        }
    )

    assert configuration.limits.max_requests is None


def test_b0_and_o1_rollout_3_configs_use_identical_resource_budgets() -> None:
    configurations = [
        load_experiment_configuration(path, project_root=ROOT)
        for path in (PILOT_CONFIG, O1_CONFIG, ACTIVE_B0_CONFIG, ACTIVE_O1_CONFIG)
    ]

    expected = ExperimentLimits(max_actions=20, max_requests=24, timeout_seconds=300)
    assert [configuration.config.limits for configuration in configurations] == [
        expected,
        expected,
        expected,
        expected,
    ]


def test_pilot_tasks_are_loaded_in_chronological_order_without_research_fields() -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    tasks = load_tasks(
        configuration.task_manifest_path,
        problem_statements_path=configuration.task_problems_path,
    )

    assert len(tasks) == 15
    assert [task.chronological_index for task in tasks] == list(range(1, 16))
    assert tasks[0].id == "GS-T001"
    prompt = build_task_prompt(tasks[0])
    assert tasks[0].problem_statement in prompt
    assert tasks[0].family_id not in prompt
    assert "recovery_pattern" not in prompt
    assert "expected_patch" not in prompt


def test_b0_runs_multiple_tasks_with_isolated_context_and_artifacts(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    model_prompts: list[str] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        user_text = "\n".join(
            str(part.content)
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        model_prompts.append(user_text)
        return ModelResponse(parts=[TextPart("offline complete")])

    def agent_factory(
        settings: Settings,
        capabilities: Sequence[AgentCapability[AgentDependencies]],
    ) -> Agent[AgentDependencies, str]:
        return create_coding_agent(
            settings,
            model=FunctionModel(respond),
            capabilities=capabilities,
        )

    workspace_root = tmp_path / "workspaces"

    def workspace_resolver(task: Task) -> Path:
        workspace = workspace_root / task.id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    store = ExperimentRunArtifactStore(tmp_path / "results")
    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        agent_factory=agent_factory,
        workspace_resolver=workspace_resolver,
        artifact_store=store,
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
    )
    tasks = [make_task("GS-T002", 2), make_task("GS-T001", 1)]

    executions = runner.run_all(tasks)

    assert [execution.artifact.task_id for execution in executions] == ["GS-T001", "GS-T002"]
    assert len({execution.artifact.run_id for execution in executions}) == 2
    assert all(execution.artifact.condition is ExperimentCondition.B0 for execution in executions)
    assert all(execution.artifact.advice_received is None for execution in executions)
    assert all(execution.artifact.retrieved_incident_id is None for execution in executions)
    assert all(execution.artifact.retrieval_score is None for execution in executions)
    assert all(execution.artifact.known_failure_repeated is False for execution in executions)
    assert all(execution.artifact_path.exists() for execution in executions)
    assert all(execution.raw_evidence_path.exists() for execution in executions)
    assert len(model_prompts) == 2
    assert "GS-T001" in model_prompts[0]
    assert "GS-T002" not in model_prompts[0]
    assert "GS-T002" in model_prompts[1]
    assert "GS-T001" not in model_prompts[1]
    assert "GS-FUTURE" not in "\n".join(model_prompts)


def test_artifact_store_is_append_only(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    store = ExperimentRunArtifactStore(tmp_path / "results")
    task = make_task("GS-T001", 1)
    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        agent_factory=lambda settings, capabilities: create_coding_agent(
            settings,
            model=FunctionModel(
                lambda _messages, _info: ModelResponse(parts=[TextPart("done")])
            ),
            capabilities=capabilities,
        ),
        workspace_resolver=lambda current_task: tmp_path / current_task.id,
        artifact_store=store,
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
    )
    execution = runner.run_task(task)

    with pytest.raises(FileExistsError):
        store.write_artifact(execution.artifact)

    restored = execution.artifact.model_validate_json(
        execution.artifact_path.read_text(encoding="utf-8")
    )
    assert restored == execution.artifact


def test_runner_wires_configured_action_timeout_and_model_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    captured: dict[str, object] = {}

    def fake_run(
        _agent: Agent[AgentDependencies, str],
        _settings: Settings,
        _dependencies: AgentDependencies,
        _prompt: str,
        **kwargs: object,
    ) -> AgentRunResult[str] | None:
        captured.update(kwargs)
        return None

    monkeypatch.setattr("graph_swarm.research.runner.run_coding_agent", fake_run)
    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        agent_factory=lambda settings, capabilities: create_coding_agent(
            settings,
            model=FunctionModel(
                lambda _messages, _info: ModelResponse(parts=[TextPart("done")])
            ),
            capabilities=capabilities,
        ),
        workspace_resolver=lambda _task: tmp_path / "workspace",
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
    )

    runner.run_task(make_task("GS-T001", 1))

    assert captured["max_actions"] == 20
    assert captured["max_requests"] == 24
    assert captured["timeout_seconds"] == 300
    assert captured["model_settings"] == {"temperature": 0}


def test_runner_passes_docker_runtime_contract_without_host_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    captured: dict[str, object] = {}

    def fake_run(
        _agent: Agent[AgentDependencies, str],
        _settings: Settings,
        dependencies: AgentDependencies,
        _prompt: str,
        **_kwargs: object,
    ) -> AgentRunResult[str] | None:
        captured["dependencies"] = dependencies
        return None

    monkeypatch.setattr("graph_swarm.research.runner.run_coding_agent", fake_run)
    runtime = ExecutionRuntime(
        runtime_type="docker",
        docker_executable=Path("docker.exe"),
        container_image="prepared:image",
        container_python_executable="/usr/bin/python3.10",
    )
    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        agent_factory=lambda settings, capabilities: create_coding_agent(
            settings,
            model=FunctionModel(
                lambda _messages, _info: ModelResponse(parts=[TextPart("done")])
            ),
            capabilities=capabilities,
        ),
        workspace_resolver=lambda _task: tmp_path / "workspace",
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
        execution_runtime_resolver=lambda _task, _workspace: runtime,
    )

    runner.run_task(make_task("GS-T001", 1))

    dependencies = captured["dependencies"]
    assert isinstance(dependencies, AgentDependencies)
    assert dependencies.execution_runtime == runtime
    assert dependencies.python_executable is None


def test_step_persistence_is_per_run_and_never_reused_as_history(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    model_prompts: list[str] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        model_prompts.append(
            "\n".join(
                str(part.content)
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart)
            )
        )
        return ModelResponse(parts=[TextPart("done")])

    def agent_factory(
        settings: Settings,
        capabilities: Sequence[AgentCapability[AgentDependencies]],
    ) -> Agent[AgentDependencies, str]:
        return create_coding_agent(
            settings,
            model=FunctionModel(respond),
            capabilities=capabilities,
        )

    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        agent_factory=agent_factory,
        workspace_resolver=lambda task: tmp_path / "workspaces" / task.id,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
    )
    executions = runner.run_all([make_task("GS-T002", 2), make_task("GS-T001", 1)])

    assert len({execution.step_database_path for execution in executions}) == 2
    assert len(model_prompts) == 2
    for execution in executions:
        with sqlite3.connect(execution.step_database_path) as connection:
            persisted_run_ids = connection.execute("SELECT run_id FROM runs").fetchall()
        assert persisted_run_ids == [(execution.artifact.run_id,)]
    assert "GS-T001" not in model_prompts[1]


def test_baseline_workspace_manager_materializes_isolated_copies(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baselines"
    baseline = baseline_root / "offline-repository"
    baseline.mkdir(parents=True)
    (baseline / "source.txt").write_text("baseline", encoding="utf-8")
    manager = BaselineWorkspaceManager(
        baseline_root,
        tmp_path / "executions",
        experiment_id="CUSTOM",
    )
    task = make_task("GS-T001", 1)

    first = manager.materialize(task, "run-1")
    (first / "source.txt").write_text("edited", encoding="utf-8")
    second = manager.materialize(task, "run-2")

    assert first != second
    assert (second / "source.txt").read_text(encoding="utf-8") == "baseline"
    assert (baseline / "source.txt").read_text(encoding="utf-8") == "baseline"


def test_objective_evaluator_does_not_promote_agent_test_event_to_success(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    result = ActionResult(
        action_id="action-1",
        tool_name="run_tests",
        success=True,
        exit_code=0,
        started_at=now,
        completed_at=now,
    )
    event = AgentEvent(
        event_id="event-1",
        run_id="run-1",
        task_id="GS-T001",
        action_id="action-1",
        event_type=AgentEventType.ACTION_COMPLETED,
        result=result,
        occurred_at=now,
    )
    evaluation = BenchmarkEvaluation(
        "GS-E003",
        lambda _task, _workspace: False,
    )

    assert evaluation.evaluate(
        make_task("GS-T001", 1),
        tmp_path,
        [event],
        None,
        None,
        0.0,
    ) is False


def test_recurrence_requires_observed_failure_evidence() -> None:
    first = BenchmarkTaskCase(make_task("GS-T001", 1), occurrence_index=1)
    later = BenchmarkTaskCase(make_task("GS-T006", 6), occurrence_index=2)
    now = datetime.now(UTC)
    failed_event = AgentEvent(
        event_id="event-1",
        run_id="run-1",
        task_id="GS-T006",
        action_id="action-1",
        event_type=AgentEventType.ACTION_COMPLETED,
        result=ActionResult(
            action_id="action-1",
            tool_name="run_tests",
            success=False,
            exit_code=1,
            started_at=now,
            completed_at=now,
        ),
        occurred_at=now,
    )

    assert benchmark_recurrence_determination(first, [], None, Path(".")) is False
    assert benchmark_recurrence_determination(later, [], None, Path(".")) is None
    assert benchmark_recurrence_determination(later, [failed_event], None, Path(".")) is None
    assert benchmark_recurrence_determination(
        BenchmarkTaskCase(make_task("synthetic", 99), occurrence_index=-1),
        [],
        None,
        Path("."),
    ) is None


def test_two_occurrence_two_runs_follow_observed_recurrence_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    outcomes = iter(
        [
            (False, "historical failure"),
            (False, "unrelated failure"),
            (True, None),
        ]
    )

    def fake_run(
        _agent: Agent[AgentDependencies, str],
        _settings: Settings,
        dependencies: AgentDependencies,
        _prompt: str,
        **_kwargs: object,
    ) -> None:
        now = datetime.now(UTC)
        action_id = f"{dependencies.run_id}-test"
        success, output = next(outcomes)
        dependencies.events.append(
            AgentEvent(
                event_id=f"{action_id}-event",
                run_id=dependencies.run_id,
                task_id=dependencies.task_id,
                action_id=action_id,
                event_type=AgentEventType.ACTION_COMPLETED,
                result=ActionResult(
                    action_id=action_id,
                    tool_name="run_tests",
                    success=success,
                    exit_code=0 if success else 1,
                    output=output,
                    started_at=now,
                    completed_at=now,
                ),
                occurred_at=now,
            )
        )
        return None

    monkeypatch.setattr("graph_swarm.research.runner.run_coding_agent", fake_run)

    def recurrence_matcher(
        _case: BenchmarkTaskCase,
        events: Sequence[AgentEvent],
        _result: AgentRunResult[str] | None,
        _workspace: Path,
    ) -> bool:
        historical_failure_type = FailureType.TEST_FAILURE
        historical_signature = "run_tests:exit_code=1"
        historical_symptom = "historical failure"
        return any(
            (failure := detect_failure(event)) is not None
            and failure.failure_type is historical_failure_type
            and failure.signature == historical_signature
            and failure.symptom == historical_symptom
            for event in events
        )

    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        agent_factory=lambda settings, capabilities: create_coding_agent(
            settings,
            model=FunctionModel(
                lambda _messages, _info: ModelResponse(parts=[TextPart("unused")])
            ),
            capabilities=capabilities,
        ),
        workspace_resolver=lambda task: tmp_path / "workspaces" / task.id,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=recurrence_matcher,
    )
    case = BenchmarkTaskCase(make_task("GS-T006", 6), occurrence_index=2)

    repeated = runner.run_case(case)
    unrelated = runner.run_case(case)
    avoided = runner.run_case(case)

    assert repeated.artifact.known_failure_repeated is True
    assert unrelated.artifact.known_failure_repeated is False
    assert avoided.artifact.known_failure_repeated is False


def test_transfer_without_explicit_recurrence_matcher_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)

    def fake_run(
        _agent: Agent[AgentDependencies, str],
        _settings: Settings,
        _dependencies: AgentDependencies,
        _prompt: str,
        **_kwargs: object,
    ) -> None:
        return None

    monkeypatch.setattr("graph_swarm.research.runner.run_coding_agent", fake_run)
    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        model=FunctionModel(
            lambda _messages, _info: ModelResponse(parts=[TextPart("unused")])
        ),
        workspace_resolver=lambda task: tmp_path / "workspaces" / task.id,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
    )

    with pytest.raises(RecurrenceEvaluationRequired):
        runner.run_case(
            BenchmarkTaskCase(make_task("GS-T006", 6), occurrence_index=2)
        )


def test_runner_rejects_unknown_recurrence_before_writing_artifact(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    runner = ExperimentRunner(
        configuration,
        settings=make_settings(),
        workspace_resolver=lambda _task: tmp_path / "workspace",
        objective_evaluator=lambda _task, _workspace: True,
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
    )

    with pytest.raises(RecurrenceEvaluationRequired):
        runner.run_case(
            BenchmarkTaskCase(make_task("synthetic", 99), occurrence_index=-1)
        )


def test_run_identity_uses_configured_experiment_id(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(PILOT_CONFIG, project_root=ROOT)
    custom = dataclasses.replace(
        configuration,
        config=configuration.config.model_copy(update={"experiment_id": "CUSTOM-EVAL"}),
    )
    runner = ExperimentRunner(
        custom,
        settings=make_settings(),
        model=FunctionModel(
            lambda _messages, _info: ModelResponse(parts=[TextPart("done")])
        ),
        workspace_resolver=lambda _task: tmp_path / "workspace",
        artifact_store=ExperimentRunArtifactStore(tmp_path / "results"),
        objective_evaluator=lambda _task, _workspace: True,
        recurrence_evaluator=lambda _case, _events, _result, _workspace: False,
    )

    execution = runner.run_task(make_task("GS-T001", 1))

    assert execution.artifact.run_id.startswith("CUSTOM-EVAL-B0-")
    assert execution.artifact.experiment_id == "CUSTOM-EVAL"
