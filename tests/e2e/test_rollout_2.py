"""GS-E002: advice before a related action enables a controlled recurrence prevention."""

import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from graph_swarm.advisory.service import AdvisoryService
from graph_swarm.agent.coding_agent import create_coding_agent, run_coding_agent
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.read_file import read_file
from graph_swarm.agent.tools.run_tests import run_tests
from graph_swarm.agent.tools.write_file import write_file
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.events import AgentEvent
from graph_swarm.domain.failures import FailureEpisode
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.repository import OperationalMemoryRepository
from graph_swarm.integration.event_persistence import persist_agent_event_stream
from graph_swarm.retrieval.candidates import (
    HistoricalActionContext,
    HistoricalRecoveryCandidate,
)
from graph_swarm.settings import Settings

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "fixtures"
    / "repositories"
    / "rollout1_agent_smoke"
)


def make_settings() -> Settings:
    return Settings(
        neo4j_uri="neo4j+s://example.databases.neo4j.io",
        neo4j_username="example-user",
        neo4j_password="example-password",
        neo4j_database="example-db",
        agent_request_limit=8,
        agent_tests_timeout_seconds=30,
    )


def make_task(task_id: str, index: int) -> Task:
    return Task(
        id=task_id,
        family_id="GS-E002-FAMILY",
        repository="rollout1_agent_smoke",
        chronological_index=index,
    )


def make_environment(environment_id: str) -> EnvironmentContext:
    return EnvironmentContext(
        id=environment_id,
        repository="rollout1_agent_smoke",
        runtime=platform.python_implementation(),
        versions={"python": platform.python_version()},
    )


def make_transfer_model() -> FunctionModel:
    corrected_source = "def add(a, b):\n    return a + b\n"

    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        parts = [part for message in messages for part in message.parts]
        tool_returns = [part for part in parts if isinstance(part, ToolReturnPart)]
        if tool_returns:
            latest_return = tool_returns[-1]
            if latest_return.tool_name == "write_file":
                return ModelResponse(parts=[ToolCallPart("run_tests", {})])
            if latest_return.tool_name == "run_tests":
                return ModelResponse(parts=[TextPart("transfer objective succeeded")])
        if any(isinstance(part, RetryPromptPart) for part in parts):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "write_file",
                        {"path": "calculator.py", "content": corrected_source},
                    )
                ]
            )
        return ModelResponse(parts=[ToolCallPart("run_tests", {})])

    return FunctionModel(model)


def make_bad_path_model() -> FunctionModel:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == "run_tests"
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("control complete")])
        return ModelResponse(parts=[ToolCallPart("run_tests", {})])

    return FunctionModel(model)


def make_candidate(
    failure: FailureEpisode,
    resolution: Resolution,
    outcome: Outcome,
    environment: EnvironmentContext,
    action: AgentEvent,
) -> HistoricalRecoveryCandidate:
    return HistoricalRecoveryCandidate(
        failure=failure,
        failed_action=HistoricalActionContext(
            id=failure.action_id,
            source_run_id=action.run_id,
            tool=action.result.tool_name,
            operation=action.result.tool_name,
            planned_at=action.result.started_at,
        ),
        environment=environment,
        resolution=resolution,
        outcomes=(outcome,),
    )


def test_gs_e002_prevents_known_recurrence(tmp_path: Path) -> None:
    acquisition_workspace = tmp_path / "acquisition"
    transfer_workspace = tmp_path / "transfer"
    control_workspace = tmp_path / "control"
    shutil.copytree(FIXTURE_PATH, acquisition_workspace)
    shutil.copytree(FIXTURE_PATH, transfer_workspace)
    shutil.copytree(FIXTURE_PATH, control_workspace)

    repository = Mock(spec=OperationalMemoryRepository)
    settings = make_settings()
    acquisition_task = make_task("task-acquisition", 1)
    acquisition_run = Run(
        id="run-acquisition",
        task_id=acquisition_task.id,
        started_at=datetime.now(UTC),
    )
    acquisition_environment = make_environment("environment-acquisition")
    acquisition_dependencies = AgentDependencies(
        acquisition_workspace,
        acquisition_run.id,
        acquisition_task.id,
    )

    first_result = run_tests(acquisition_dependencies, timeout_seconds=30)
    read_result = read_file(acquisition_dependencies, "calculator.py")
    assert read_result.output is not None
    write_result = write_file(
        acquisition_dependencies,
        "calculator.py",
        read_result.output.replace("return a - b", "return a + b"),
    )
    final_result = run_tests(acquisition_dependencies, timeout_seconds=30)
    assert first_result.success is False
    assert write_result.success is True
    assert final_result.success is True

    acquisition_chain = persist_agent_event_stream(
        repository,
        acquisition_dependencies.events,
        acquisition_task,
        acquisition_run,
        acquisition_environment,
    )
    assert acquisition_chain is not None
    failure, resolution, outcome = acquisition_chain
    assert isinstance(failure, FailureEpisode)
    assert isinstance(resolution, Resolution)
    assert isinstance(outcome, Outcome)
    assert resolution.status.value == "observed_successful"
    assert outcome.success is True
    assert any(call.args[0].id == failure.id for call in repository.save_failure.call_args_list)
    assert any(
        call.args[0].id == resolution.id
        for call in repository.save_resolution.call_args_list
    )
    assert any(call.args[0].id == outcome.id for call in repository.save_outcome.call_args_list)

    historical_candidate = make_candidate(
        failure,
        resolution,
        outcome,
        acquisition_environment,
        acquisition_dependencies.events[0],
    )
    repository.find_historical_recovery_candidates.return_value = (historical_candidate,)

    transfer_task = make_task("task-transfer", 2)
    transfer_run = Run(
        id="run-transfer",
        task_id=transfer_task.id,
        started_at=acquisition_run.started_at,
    )
    transfer_environment = make_environment("environment-transfer")
    transfer_dependencies = AgentDependencies(
        transfer_workspace,
        transfer_run.id,
        transfer_task.id,
        task=transfer_task,
        environment=transfer_environment,
        advisory_service=AdvisoryService(repository),
    )
    transfer_agent = create_coding_agent(settings, model=make_transfer_model())

    transfer_result = run_coding_agent(
        transfer_agent,
        settings,
        transfer_dependencies,
        "Repair the failing calculator behavior and verify it.",
    )

    assert transfer_result.output == "transfer objective succeeded"
    assert repository.find_historical_recovery_candidates.call_count >= 1
    assert len(transfer_dependencies.advice_events) == 1
    advice_event = transfer_dependencies.advice_events[0]
    assert advice_event.issued_at < transfer_dependencies.events[0].occurred_at
    assert advice_event.advice.provenance.failure_episode_id == failure.id
    assert advice_event.advice.provenance.resolution_id == resolution.id
    assert advice_event.planned_action.tool == "run_tests"
    assert historical_candidate.failed_action.source_run_id == acquisition_run.id
    assert advice_event.planned_action.run_id == transfer_run.id
    assert transfer_dependencies.behavior_evidence[0].observation == "changed"
    assert transfer_dependencies.behavior_evidence[0].actual_action_after_advice is not None
    assert (
        transfer_dependencies.behavior_evidence[0].actual_action_after_advice.tool
        == "write_file"
    )
    assert all(event.result.success for event in transfer_dependencies.events)
    assert all(
        event.result.tool_name != "run_tests" or event.result.success
        for event in transfer_dependencies.events
    )
    assert "return a + b" in (transfer_workspace / "calculator.py").read_text(encoding="utf-8")

    control_task = make_task("task-control", 2)
    control_run = Run(
        id="run-control",
        task_id=control_task.id,
        started_at=acquisition_run.started_at,
    )
    control_dependencies = AgentDependencies(
        control_workspace,
        control_run.id,
        control_task.id,
    )
    control_agent = create_coding_agent(settings, model=make_bad_path_model())
    run_coding_agent(
        control_agent,
        settings,
        control_dependencies,
        "Verify the calculator behavior.",
    )

    assert len(control_dependencies.advice_events) == 0
    assert len(control_dependencies.events) == 1
    assert control_dependencies.events[0].result.tool_name == "run_tests"
    assert control_dependencies.events[0].result.success is False
