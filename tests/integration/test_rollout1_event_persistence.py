"""Persistence integration for the deterministic Rollout 1 event stream."""

import os
import platform
import shutil
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest

from graph_swarm.agent.dependencies import AgentDependencies
from graph_swarm.agent.tools.read_file import read_file
from graph_swarm.agent.tools.run_tests import run_tests
from graph_swarm.agent.tools.write_file import write_file
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.resolutions import ResolutionStatus
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
    / "benchmarks"
    / "fixtures"
    / "repositories"
    / "rollout1_agent_smoke"
)
TASK_ID = "GS-E001-LOCAL"
RUN_ID = f"{TASK_ID}-RUN"
ENVIRONMENT_ID = f"{TASK_ID}-ENV"


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


def _count(repository: Neo4jRepository, query: str, **parameters: object) -> int:
    result = repository.execute_query(query, **parameters)
    return int(result.records[0]["count"])


def _cleanup(
    repository: Neo4jRepository,
    action_ids: list[str],
    failure_id: str | None,
    resolution_id: str | None,
    outcome_id: str | None,
) -> None:
    node_ids = [TASK_ID, RUN_ID, ENVIRONMENT_ID, *action_ids]
    for node_id in (failure_id, resolution_id, outcome_id):
        if node_id is not None:
            node_ids.append(node_id)

    repository.execute_query(
        """
        MATCH (n)
        WHERE n.id IN $node_ids
        DETACH DELETE n
        """,
        node_ids=node_ids,
    )


def test_event_stream_persists_execution_and_one_failure(
    tmp_path: Path,
    repository: Neo4jRepository,
) -> None:
    copied_fixture = tmp_path / "rollout1_agent_smoke"
    shutil.copytree(FIXTURE_PATH, copied_fixture)
    dependencies = AgentDependencies(copied_fixture, RUN_ID, TASK_ID)
    task = Task(
        id=TASK_ID,
        family_id="GS-E001",
        repository="rollout1_agent_smoke",
        chronological_index=1,
    )
    run = Run(id=RUN_ID, task_id=TASK_ID, started_at=datetime.now(UTC))
    environment = EnvironmentContext(
        id=ENVIRONMENT_ID,
        repository="rollout1_agent_smoke",
        runtime=platform.python_implementation(),
        versions={"python": platform.python_version()},
    )
    failure_id: str | None = None
    resolution_id: str | None = None
    outcome_id: str | None = None
    action_ids: list[str] = []
    reconnected_repository: Neo4jRepository | None = None

    try:
        first_test_result = run_tests(dependencies, timeout_seconds=30)
        assert first_test_result.success is False
        assert first_test_result.exit_code is not None

        read_result = read_file(dependencies, "calculator.py")
        assert read_result.success is True
        assert read_result.output is not None

        write_result = write_file(
            dependencies,
            "calculator.py",
            read_result.output.replace("return a - b", "return a + b"),
        )
        assert write_result.success is True
        assert write_result.output is not None

        final_test_result = run_tests(dependencies, timeout_seconds=30)
        assert final_test_result.success is True
        assert final_test_result.exit_code == 0
        assert len(dependencies.events) == 4
        action_ids = [event.action_id for event in dependencies.events]

        persisted_chain = persist_agent_event_stream(
            repository,
            dependencies.events,
            task,
            run,
            environment,
        )
        assert persisted_chain is not None
        failure, resolution, outcome = persisted_chain
        failure_id = failure.id
        resolution_id = resolution.id
        outcome_id = outcome.id
        repeated_chain = persist_agent_event_stream(
            repository,
            dependencies.events,
            task,
            run,
            environment,
        )
        assert repeated_chain is not None
        assert repeated_chain[0].id == failure.id
        assert repeated_chain[1].id == resolution.id
        assert repeated_chain[2].id == outcome.id
        assert _count(
            repository,
            """
            MATCH (action:Action)
            WHERE action.id IN $action_ids
            RETURN count(action) AS count
            """,
            action_ids=action_ids,
        ) == 4
        assert _count(
            repository,
            """
            MATCH (tool:Tool)
            WHERE tool.name IN $tool_names
            RETURN count(tool) AS count
            """,
            tool_names=["run_tests", "read_file", "write_file"],
        ) == 3
        assert _count(
            repository,
            "MATCH (task:Task {id: $id}) RETURN count(task) AS count",
            id=TASK_ID,
        ) == 1
        assert _count(
            repository,
            "MATCH (run:Run {id: $id}) RETURN count(run) AS count",
            id=RUN_ID,
        ) == 1
        assert _count(
            repository,
            "MATCH (environment:Environment {id: $id}) RETURN count(environment) AS count",
            id=ENVIRONMENT_ID,
        ) == 1
        assert _count(
            repository,
            "MATCH (resolution:Resolution {id: $id}) RETURN count(resolution) AS count",
            id=resolution.id,
        ) == 1
        assert _count(
            repository,
            "MATCH (outcome:Outcome {id: $id}) RETURN count(outcome) AS count",
            id=outcome.id,
        ) == 1
        assert _count(
            repository,
            """
            MATCH (failure:FailureEpisode)-[:OCCURRED_IN]->
                (environment:Environment {id: $id})
            RETURN count(failure) AS count
            """,
            id=ENVIRONMENT_ID,
        ) == 1
        assert _count(
            repository,
            """
            MATCH (action:Action {id: $action_id})
            MATCH (failure:FailureEpisode {id: $failure_id})
            MATCH (action)-[r:PART_OF_FAILURE]->(failure)
            RETURN count(r) AS count
            """,
            action_id=dependencies.events[0].action_id,
            failure_id=failure.id,
        ) == 1
        assert _count(
            repository,
            """
            MATCH (failure:FailureEpisode {id: $failure_id})
            MATCH (resolution:Resolution {id: $resolution_id})
            MATCH (failure)-[r:RESOLVED_BY]->(resolution)
            RETURN count(r) AS count
            """,
            failure_id=failure.id,
            resolution_id=resolution.id,
        ) == 1
        assert _count(
            repository,
            """
            MATCH (resolution:Resolution {id: $resolution_id})
            MATCH (outcome:Outcome {id: $outcome_id})
            MATCH (resolution)-[r:VERIFIED_BY]->(outcome)
            RETURN count(r) AS count
            """,
            resolution_id=resolution.id,
            outcome_id=outcome.id,
        ) == 1

        settings = get_settings()
        candidate_repository = Neo4jRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        try:
            candidate_repository.verify_connectivity()
        except Exception:
            candidate_repository.close()
            raise
        reconnected_repository = candidate_repository
        repository.close()

        failed_event = dependencies.events[0]
        incident = reconnected_repository.get_incident_lineage(failure.id)
        assert incident.task.id == TASK_ID
        assert incident.run.id == RUN_ID
        assert incident.environment.id == ENVIRONMENT_ID
        assert incident.failure.id == failure.id
        assert {action.planned_action.id for action in incident.actions} == set(action_ids)
        assert {tool.name for tool in incident.tools} == {
            "run_tests",
            "read_file",
            "write_file",
        }
        assert len(incident.resolutions) == 1
        assert incident.resolutions[0].id == resolution.id
        assert incident.resolutions[0].status is ResolutionStatus.OBSERVED_SUCCESSFUL
        assert incident.resolutions[0].successful_observations == 1
        assert dependencies.events[2].action_id in incident.resolutions[0].description
        assert write_result.output in incident.resolutions[0].description
        assert dependencies.events[3].action_id in incident.resolutions[0].description
        assert len(incident.outcomes) == 1
        assert incident.outcomes[0].id == outcome.id
        assert incident.outcomes[0].action_id == dependencies.events[3].action_id
        assert incident.outcomes[0].success is True
        assert incident.outcomes[0].exit_code == 0
        failed_action = next(
            action
            for action in incident.actions
            if action.planned_action.id == failed_event.action_id
        )
        assert failed_action.result.action_id == failed_event.action_id
        assert "return a - b" in (FIXTURE_PATH / "calculator.py").read_text(encoding="utf-8")
    finally:
        cleanup_repository = reconnected_repository or repository
        try:
            _cleanup(cleanup_repository, action_ids, failure_id, resolution_id, outcome_id)
        finally:
            if reconnected_repository is not None:
                reconnected_repository.close()
