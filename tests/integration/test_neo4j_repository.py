"""Real Neo4j persistence and idempotency integration tests."""

import os
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import import_module
from typing import cast
from uuid import uuid4

import pytest

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph.neo4j_repository import Neo4jRepository
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

STARTED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
COMPLETED_AT = STARTED_AT + timedelta(seconds=1)


@dataclass(frozen=True)
class IncidentFixture:
    prefix: str
    run: Run
    task: Task
    action: PlannedAction
    result: ActionResult
    tool: Tool
    environment: EnvironmentContext
    failure: FailureEpisode
    resolution: Resolution
    outcome: Outcome


def make_fixture() -> IncidentFixture:
    prefix = f"integration-{uuid4().hex}"
    return IncidentFixture(
        prefix=prefix,
        run=Run(id=f"{prefix}-run", task_id=f"{prefix}-task", started_at=STARTED_AT),
        task=Task(
            id=f"{prefix}-task",
            family_id=f"{prefix}-family",
            repository="integration/example",
            chronological_index=1,
        ),
        action=PlannedAction(
            id=f"{prefix}-action",
            run_id=f"{prefix}-run",
            task_id=f"{prefix}-task",
            tool=f"{prefix}-tool",
            operation="pytest",
            arguments={"paths": ["tests"], "options": {"quiet": True}},
            planned_at=STARTED_AT,
        ),
        result=ActionResult(
            action_id=f"{prefix}-action",
            tool_name=f"{prefix}-tool",
            success=False,
            exit_code=1,
            output="one test failed",
            error="failure fixture",
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
        ),
        tool=Tool(name=f"{prefix}-tool"),
        environment=EnvironmentContext(
            id=f"{prefix}-environment",
            repository="integration/example",
            runtime="python-3.13",
            versions={"pytest": "8"},
            markers={"ci": "integration"},
        ),
        failure=FailureEpisode(
            id=f"{prefix}-failure",
            action_id=f"{prefix}-action",
            failure_type=FailureType.TEST_FAILURE,
            signature="integration-signature",
            symptom="one test failed",
            observed_at=COMPLETED_AT,
        ),
        resolution=Resolution(
            id=f"{prefix}-resolution",
            failure_id=f"{prefix}-failure",
            description="Apply the verified fixture resolution",
            status=ResolutionStatus.OBSERVED_SUCCESSFUL,
            successful_observations=1,
        ),
        outcome=Outcome(
            id=f"{prefix}-outcome",
            action_id=f"{prefix}-action",
            success=True,
            tests_passed=1,
            tests_failed=0,
            exit_code=0,
            observed_at=COMPLETED_AT,
        ),
    )


def persist_fixture(repository: Neo4jRepository, fixture: IncidentFixture) -> None:
    repository.save_run(fixture.run)
    repository.save_task(fixture.task)
    repository.save_action(fixture.action, fixture.result)
    repository.save_tool(fixture.tool)
    repository.save_environment(fixture.environment)
    repository.save_failure(fixture.failure)
    repository.save_resolution(fixture.resolution)
    repository.save_outcome(fixture.outcome)
    repository.link_task_action(fixture.task.id, fixture.action.id)
    repository.link_action_tool(fixture.action.id, fixture.tool.name)
    repository.link_action_run(fixture.action.id, fixture.run.id)
    repository.link_action_failure(fixture.action.id, fixture.failure.id)
    repository.link_failure_environment(fixture.failure.id, fixture.environment.id)
    repository.link_failure_resolution(fixture.failure.id, fixture.resolution.id)
    repository.link_resolution_outcome(fixture.resolution.id, fixture.outcome.id)


def cleanup_fixture(
    repository: Neo4jRepository,
    prefix: str,
    failure_id: str | None = None,
) -> None:
    repository.execute_query(
        """
        MATCH (n)
        WHERE n.id STARTS WITH $prefix
           OR n.name STARTS WITH $prefix
           OR n.id = $failure_id
        DETACH DELETE n
        """,
        prefix=prefix,
        failure_id=failure_id,
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


def count_node(repository: Neo4jRepository, label: str, property_name: str, value: str) -> int:
    result = repository.execute_query(
        f"MATCH (n:{label} {{{property_name}: $value}}) RETURN count(n) AS count",
        value=value,
    )
    return int(result.records[0]["count"])


def count_relationship(
    repository: Neo4jRepository,
    source_label: str,
    source_property: str,
    source_value: str,
    relationship: str,
    target_label: str,
    target_property: str,
    target_value: str,
) -> int:
    result = repository.execute_query(
        f"""
        MATCH (source:{source_label} {{{source_property}: $source_value}})
        MATCH (target:{target_label} {{{target_property}: $target_value}})
        MATCH (source)-[r:{relationship}]->(target)
        RETURN count(r) AS count
        """,
        source_value=source_value,
        target_value=target_value,
    )
    return int(result.records[0]["count"])


def test_complete_incident_lineage_and_idempotency(repository: Neo4jRepository) -> None:
    fixture = make_fixture()
    try:
        persist_fixture(repository, fixture)
        persist_fixture(repository, fixture)

        incident = repository.get_incident_lineage(fixture.failure.id)

        assert incident.failure.id == fixture.failure.id
        assert incident.run.id == fixture.run.id
        assert incident.task.id == fixture.task.id
        assert len(incident.actions) == 1
        assert incident.actions[0].planned_action.id == fixture.action.id
        assert incident.actions[0].result.action_id == fixture.action.id
        assert incident.tools[0].name == fixture.tool.name
        assert incident.environment.id == fixture.environment.id
        assert incident.resolutions[0].id == fixture.resolution.id
        assert incident.outcomes[0].id == fixture.outcome.id

        candidates = repository.find_historical_recovery_candidates(
            fixture.action,
            fixture.environment,
        )
        assert len(candidates) == 1
        assert candidates[0].failure.id == fixture.failure.id
        assert candidates[0].resolution.id == fixture.resolution.id
        assert candidates[0].outcomes[0].id == fixture.outcome.id

        expected_nodes = (
            ("Run", "id", fixture.run.id),
            ("Task", "id", fixture.task.id),
            ("Action", "id", fixture.action.id),
            ("Tool", "name", fixture.tool.name),
            ("Environment", "id", fixture.environment.id),
            ("FailureEpisode", "id", fixture.failure.id),
            ("Resolution", "id", fixture.resolution.id),
            ("Outcome", "id", fixture.outcome.id),
        )
        for label, property_name, value in expected_nodes:
            assert count_node(repository, label, property_name, value) == 1

        expected_relationships = (
            ("Task", "id", fixture.task.id, "HAS_ACTION", "Action", "id", fixture.action.id),
            ("Action", "id", fixture.action.id, "USED", "Tool", "name", fixture.tool.name),
            ("Action", "id", fixture.action.id, "PART_OF", "Run", "id", fixture.run.id),
            (
                "Action",
                "id",
                fixture.action.id,
                "PART_OF_FAILURE",
                "FailureEpisode",
                "id",
                fixture.failure.id,
            ),
            (
                "FailureEpisode",
                "id",
                fixture.failure.id,
                "OCCURRED_IN",
                "Environment",
                "id",
                fixture.environment.id,
            ),
            (
                "FailureEpisode",
                "id",
                fixture.failure.id,
                "RESOLVED_BY",
                "Resolution",
                "id",
                fixture.resolution.id,
            ),
            (
                "Resolution",
                "id",
                fixture.resolution.id,
                "VERIFIED_BY",
                "Outcome",
                "id",
                fixture.outcome.id,
            ),
        )
        for relationship in expected_relationships:
            assert count_relationship(repository, *relationship) == 1
    finally:
        cleanup_fixture(repository, fixture.prefix)
