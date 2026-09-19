"""Live Neo4j integration for the deterministic Rollout 2 advisory path."""

import os
import platform
import shutil
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from pydantic_ai import ModelRetry

from graph_swarm.advisory.service import AdvisoryService
from graph_swarm.agent.advisory import prepare_tool_action
from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.read_file import read_file
from graph_swarm.agent.tools.run_tests import run_tests
from graph_swarm.agent.tools.write_file import write_file
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.graph.neo4j_repository import Neo4jRepository
from graph_swarm.integration.event_persistence import persist_agent_event_stream
from graph_swarm.settings import get_settings
from scripts.setup_neo4j import apply_schema

RUN_INTEGRATION = os.getenv("GRAPH_SWARM_RUN_NEO4J_INTEGRATION") == "1"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason="Set GRAPH_SWARM_RUN_NEO4J_INTEGRATION=1 to run against Neo4j",
    ),
]

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "fixtures"
    / "repositories"
    / "rollout1_agent_smoke"
)


@pytest.fixture
def repository() -> Generator[Neo4jRepository, None, None]:
    settings = get_settings()
    if settings.neo4j_uri.startswith("neo4j+s://"):
        truststore = import_module("truststore")
        cast(Callable[[], None], truststore.__dict__["inject_into_ssl"])()

    repository = Neo4jRepository(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    repository.verify_connectivity()
    apply_schema(repository)
    try:
        yield repository
    finally:
        repository.close()


def _cleanup(repository: Neo4jRepository, node_ids: list[str]) -> None:
    repository.execute_query(
        """
        MATCH (node)
        WHERE node.id IN $node_ids
        DETACH DELETE node
        """,
        node_ids=node_ids,
    )


def test_gs_e002_graph_backed_advisory_path(
    tmp_path: Path,
    repository: Neo4jRepository,
) -> None:
    prefix = f"GS-E002-NEO4J-{uuid4().hex}"
    acquisition_task = Task(
        id=f"{prefix}-ACQUISITION-TASK",
        problem_statement="Fix the repository so its tests pass.",
        family_id="GS-E002-FAMILY",
        repository="rollout1_agent_smoke",
        chronological_index=1,
    )
    acquisition_run = Run(
        id=f"{prefix}-ACQUISITION-RUN",
        task_id=acquisition_task.id,
        started_at=datetime.now(UTC),
    )
    acquisition_environment = EnvironmentContext(
        id=f"{prefix}-ACQUISITION-ENVIRONMENT",
        repository="rollout1_agent_smoke",
        runtime=platform.python_implementation(),
        versions={"python": platform.python_version()},
    )
    acquisition_workspace = tmp_path / "acquisition"
    shutil.copytree(FIXTURE_PATH, acquisition_workspace)
    acquisition_dependencies = AgentDependencies(
        acquisition_workspace,
        acquisition_run.id,
        acquisition_task.id,
    )
    created_node_ids = [
        acquisition_task.id,
        acquisition_run.id,
        acquisition_environment.id,
    ]

    try:
        first_action = PlannedAction(
            id=f"{prefix}-ACQUISITION-ACTION-TEST-BEFORE",
            run_id=acquisition_run.id,
            task_id=acquisition_task.id,
            tool="run_tests",
            operation="run_tests",
            planned_at=datetime.now(UTC),
        )
        first_result = run_tests(
            acquisition_dependencies,
            timeout_seconds=30,
            action_id=first_action.id,
        )
        read_action = PlannedAction(
            id=f"{prefix}-ACQUISITION-ACTION-READ",
            run_id=acquisition_run.id,
            task_id=acquisition_task.id,
            tool="read_file",
            operation="read_file",
            arguments={"path": "calculator.py"},
            planned_at=datetime.now(UTC),
        )
        read_result = read_file(
            acquisition_dependencies,
            "calculator.py",
            action_id=read_action.id,
        )
        assert read_result.output is not None
        corrected_source = read_result.output.replace("return a - b", "return a + b")
        write_action = PlannedAction(
            id=f"{prefix}-ACQUISITION-ACTION-WRITE",
            run_id=acquisition_run.id,
            task_id=acquisition_task.id,
            tool="write_file",
            operation="write_file",
            arguments={"path": "calculator.py", "content": corrected_source},
            planned_at=datetime.now(UTC),
        )
        write_result = write_file(
            acquisition_dependencies,
            "calculator.py",
            corrected_source,
            action_id=write_action.id,
        )
        final_action = PlannedAction(
            id=f"{prefix}-ACQUISITION-ACTION-TEST-AFTER",
            run_id=acquisition_run.id,
            task_id=acquisition_task.id,
            tool="run_tests",
            operation="run_tests",
            planned_at=datetime.now(UTC),
        )
        final_result = run_tests(
            acquisition_dependencies,
            timeout_seconds=30,
            action_id=final_action.id,
        )
        assert first_result.success is False
        assert write_result.success is True
        assert final_result.success is True
        created_node_ids.extend(
            event.action_id for event in acquisition_dependencies.events
        )

        acquisition_chain = persist_agent_event_stream(
            repository,
            acquisition_dependencies.events,
            acquisition_task,
            acquisition_run,
            acquisition_environment,
            planned_actions=(first_action, read_action, write_action, final_action),
        )
        assert acquisition_chain is not None
        failure, resolution, outcome = acquisition_chain
        created_node_ids.extend(
            [
                failure.id,
                resolution.id,
                outcome.id,
            ]
        )
        assert resolution.status.value == "observed_successful"
        assert outcome.success is True

        transfer_task = Task(
            id=f"{prefix}-TRANSFER-TASK",
            problem_statement="Fix the repository so its tests pass.",
            family_id="GS-E002-FAMILY",
            repository="rollout1_agent_smoke",
            chronological_index=2,
        )
        transfer_run = Run(
            id=f"{prefix}-TRANSFER-RUN",
            task_id=transfer_task.id,
            started_at=datetime.now(UTC),
        )
        transfer_environment = EnvironmentContext(
            id=f"{prefix}-TRANSFER-ENVIRONMENT",
            repository="rollout1_agent_smoke",
            runtime=platform.python_implementation(),
            versions={"python": platform.python_version()},
        )
        repository.save_task(transfer_task)
        repository.save_run(transfer_run)
        repository.save_environment(transfer_environment)
        created_node_ids.extend(
            [transfer_task.id, transfer_run.id, transfer_environment.id]
        )

        transfer_action = PlannedAction(
            id=f"{prefix}-TRANSFER-ACTION",
            run_id=transfer_run.id,
            task_id=transfer_task.id,
            tool="run_tests",
            operation="run_tests",
            arguments={},
            planned_at=datetime.now(UTC),
        )
        candidates = repository.find_historical_recovery_candidates(
            transfer_action,
            transfer_environment,
        )
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.failed_action.source_run_id == acquisition_run.id
        assert candidate.failed_action.source_run_id != transfer_run.id
        assert candidate.failure.id == failure.id
        assert candidate.resolution.id == resolution.id
        assert any(candidate_outcome.success for candidate_outcome in candidate.outcomes)

        advisory_service = AdvisoryService(repository)
        advice_result = advisory_service.evaluate_action(
            transfer_task,
            transfer_action,
            transfer_environment,
        )
        assert advice_result.has_advice is True
        assert advice_result.provenance is not None
        assert advice_result.provenance.failure_episode_id == failure.id
        assert advice_result.provenance.resolution_id == resolution.id
        assert advice_result.provenance.outcome_ids == (outcome.id,)

        transfer_dependencies = AgentDependencies(
            tmp_path / "transfer",
            transfer_run.id,
            transfer_task.id,
            task=transfer_task,
            environment=transfer_environment,
            advisory_service=advisory_service,
        )
        try:
            prepare_tool_action(
                transfer_dependencies,
                transfer_action.tool,
                transfer_action.operation,
                transfer_action.arguments,
            )
        except ModelRetry:
            pass
        else:
            pytest.fail("expected pre-execution advisory ModelRetry")

        assert transfer_dependencies.events == []
        assert len(transfer_dependencies.advice_events) == 1
        advice_event = transfer_dependencies.advice_events[0]
        assert advice_event.planned_action.arguments == transfer_action.arguments
        assert advice_event.planned_action.tool == transfer_action.tool
        assert advice_event.planned_action.operation == transfer_action.operation
        assert advice_event.advice.provenance.failure_episode_id == failure.id
        assert advice_event.advice.provenance.resolution_id == resolution.id

        same_run_action = transfer_action.model_copy(
            update={"run_id": acquisition_run.id, "task_id": acquisition_task.id}
        )
        same_run_result = advisory_service.evaluate_action(
            acquisition_task,
            same_run_action,
            acquisition_environment,
        )
        assert same_run_result.has_advice is False
    finally:
        _cleanup(repository, created_node_ids)
