"""Real Neo4j persistence and idempotency integration tests."""

import os
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import import_module
from math import sqrt
from typing import cast
from uuid import uuid4

import pytest

from graph_swarm.domain.action import ActionResult
from graph_swarm.domain.actions import PlannedAction
from graph_swarm.domain.environment import EnvironmentContext
from graph_swarm.domain.failures import FailureEpisode, FailureType
from graph_swarm.domain.outcomes import Outcome
from graph_swarm.domain.recovery_patterns import (
    EnvironmentConstraints,
    RecoveryPattern,
    RecoveryPatternStatus,
)
from graph_swarm.domain.resolutions import Resolution, ResolutionStatus
from graph_swarm.domain.runs import Run
from graph_swarm.domain.tasks import Task
from graph_swarm.domain.tools import Tool
from graph_swarm.graph.neo4j_repository import Neo4jRepository
from graph_swarm.memory.recovery_embeddings import RecoveryPatternEmbedder
from graph_swarm.retrieval.service import RecoveryPatternRetrievalService
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


def make_fixture(
    *,
    repository: str = "integration/example",
    tool_name: str | None = None,
    chronological_index: int = 1,
) -> IncidentFixture:
    prefix = f"integration-{uuid4().hex}"
    resolved_tool_name = tool_name or f"{prefix}-tool"
    return IncidentFixture(
        prefix=prefix,
        run=Run(id=f"{prefix}-run", task_id=f"{prefix}-task", started_at=STARTED_AT),
        task=Task(
            id=f"{prefix}-task",
            problem_statement="Fix the integration fixture so its tests pass.",
            family_id=f"{prefix}-family",
            repository=repository,
            chronological_index=chronological_index,
        ),
        action=PlannedAction(
            id=f"{prefix}-action",
            run_id=f"{prefix}-run",
            task_id=f"{prefix}-task",
            tool=resolved_tool_name,
            operation="pytest",
            arguments={"paths": ["tests"], "options": {"quiet": True}},
            planned_at=STARTED_AT,
        ),
        result=ActionResult(
            action_id=f"{prefix}-action",
            tool_name=resolved_tool_name,
            success=False,
            exit_code=1,
            output="one test failed",
            error="failure fixture",
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
        ),
        tool=Tool(name=resolved_tool_name),
        environment=EnvironmentContext(
            id=f"{prefix}-environment",
            repository=repository,
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
        assert incident.task.problem_statement == fixture.task.problem_statement
        assert len(incident.actions) == 1
        assert incident.actions[0].planned_action.id == fixture.action.id
        assert incident.actions[0].result.action_id == fixture.action.id
        assert incident.tools[0].name == fixture.tool.name
        assert incident.environment.id == fixture.environment.id
        assert incident.resolutions[0].id == fixture.resolution.id
        assert incident.outcomes[0].id == fixture.outcome.id

        future_action = fixture.action.model_copy(
            update={
                "id": f"{fixture.prefix}-future-action",
                "run_id": f"{fixture.prefix}-transfer-run",
                "planned_at": COMPLETED_AT + timedelta(seconds=1),
            }
        )
        candidates = repository.find_historical_recovery_candidates(
            future_action,
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


def test_native_recovery_pattern_vector_index_persistence_and_query(
    repository: Neo4jRepository,
) -> None:
    fixture = make_fixture()
    recovery_action = fixture.action.model_copy(
        update={
            "id": f"{fixture.prefix}-recovery-action",
            "tool": "write_file",
            "operation": "write_file",
            "arguments": {"path": "src/example.py", "content": "return value"},
        }
    )
    recovery_result = ActionResult(
        action_id=recovery_action.id,
        tool_name=recovery_action.tool,
        success=True,
        exit_code=0,
        output="recovery observed",
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
    )
    pattern = RecoveryPattern(
        id=f"{fixture.prefix}-pattern",
        title="Restore the failing assertion",
        guidance="Inspect the assertion input and apply the observed correction.",
        source_failure_id=fixture.failure.id,
        source_resolution_id=fixture.resolution.id,
        source_outcome_id=fixture.outcome.id,
        source_task_id=fixture.task.id,
        source_chronological_index=fixture.task.chronological_index,
        source_tool=fixture.action.tool,
        source_operation=fixture.action.operation,
        source_failure_type=fixture.failure.failure_type.value,
        environment_constraints=EnvironmentConstraints(runtime=fixture.environment.runtime),
        verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
        evidence_count=1,
        evidence_summary="One observed successful recovery.",
        embedding=[1.0] + [0.0] * 383,
        created_at=STARTED_AT,
    )

    try:
        persist_fixture(repository, fixture)
        repository.save_action(recovery_action, recovery_result)
        repository.save_tool(Tool(name=recovery_action.tool))
        repository.link_task_action(fixture.task.id, recovery_action.id)
        repository.link_action_tool(recovery_action.id, recovery_action.tool)
        repository.link_action_run(recovery_action.id, fixture.run.id)
        repository.link_resolution_observed_change(
            fixture.resolution.id,
            recovery_action.id,
        )
        repository.save_recovery_pattern(pattern)
        repository.ensure_recovery_pattern_vector_index()

        index_online = False
        for _ in range(30):
            index_result = repository.execute_query(
                """
                SHOW VECTOR INDEXES
                YIELD name, state
                WHERE name = 'recovery_pattern_embedding_idx'
                RETURN state
                """
            )
            if index_result.records and index_result.records[0]["state"] == "ONLINE":
                index_online = True
                break
            time.sleep(0.5)
        assert index_online

        candidates = repository.query_recovery_pattern_vectors(
            [1.0] + [0.0] * 383,
            limit=5,
        )

        assert candidates
        assert candidates[0].pattern.id == pattern.id
        assert isinstance(candidates[0].vector_score, float)
        assert candidates[0].vector_score == pytest.approx(1.0)
    finally:
        cleanup_fixture(repository, fixture.prefix)


class ConstantQueryEncoder:
    def encode(self, text: str, *, normalize_embeddings: bool) -> object:
        assert text
        assert normalize_embeddings is True
        return [1.0] + [0.0] * 383


def test_live_retrieval_filters_future_and_incompatible_patterns(
    repository: Neo4jRepository,
) -> None:
    common_tool = "sprint5-vector-tool"
    earlier = make_fixture(
        repository="repo-A",
        tool_name=common_tool,
        chronological_index=1,
    )
    future = make_fixture(
        repository="repo-A",
        tool_name=common_tool,
        chronological_index=6,
    )

    def pattern_for(
        fixture: IncidentFixture,
        pattern_id: str,
        vector_score: float,
        *,
        runtime: str = "python-3.13",
    ) -> RecoveryPattern:
        return RecoveryPattern(
            id=pattern_id,
            title="Restore the historical condition",
            guidance="Apply the observed correction.",
            source_failure_id=fixture.failure.id,
            source_resolution_id=fixture.resolution.id,
            source_outcome_id=fixture.outcome.id,
            source_task_id=fixture.task.id,
            source_chronological_index=fixture.task.chronological_index,
            source_tool=fixture.action.tool,
            source_operation=fixture.action.operation,
            source_failure_type=fixture.failure.failure_type.value,
            environment_constraints=EnvironmentConstraints(
                runtime=runtime,
                versions={"pytest": "8"},
                markers={"ci": "integration"},
            ),
            verification_status=RecoveryPatternStatus.OBSERVED_SUCCESSFUL,
            evidence_count=1,
            evidence_summary="Observed successful recovery.",
            embedding=[vector_score, sqrt(1 - vector_score**2)] + [0.0] * 382,
            created_at=STARTED_AT,
        )

    incompatible = make_fixture(
        repository="repo-A",
        tool_name=common_tool,
        chronological_index=2,
    )
    patterns = (
        pattern_for(earlier, f"{earlier.prefix}-pattern", 0.80),
        pattern_for(future, f"{future.prefix}-pattern", 0.99),
        pattern_for(
            incompatible,
            f"{incompatible.prefix}-pattern",
            0.95,
            runtime="python-3.12",
        ),
    )
    current_task = Task(
        id=f"{earlier.prefix}-current-task",
        problem_statement="Restore the historical condition.",
        family_id="must-not-enter-retrieval",
        repository="repo-B",
        chronological_index=5,
    )
    current_action = PlannedAction(
        id=f"{earlier.prefix}-current-action",
        run_id=f"{earlier.prefix}-current-run",
        task_id=current_task.id,
        tool=common_tool,
        operation="pytest",
        planned_at=STARTED_AT,
    )
    current_environment = EnvironmentContext(
        id=f"{earlier.prefix}-current-environment",
        repository="repo-B",
        runtime="python-3.13",
        versions={"pytest": "8"},
        markers={"ci": "integration"},
    )

    try:
        for fixture in (earlier, future, incompatible):
            persist_fixture(repository, fixture)
            recovery_action = fixture.action.model_copy(
                update={
                    "id": f"{fixture.prefix}-recovery-action",
                    "tool": "write_file",
                    "operation": "write_file",
                    "arguments": {"path": "src/example.py", "content": "return value"},
                }
            )
            recovery_result = ActionResult(
                action_id=recovery_action.id,
                tool_name=recovery_action.tool,
                success=True,
                exit_code=0,
                output="recovery observed",
                started_at=STARTED_AT,
                completed_at=COMPLETED_AT,
            )
            repository.save_action(recovery_action, recovery_result)
            repository.save_tool(Tool(name=recovery_action.tool))
            repository.link_task_action(fixture.task.id, recovery_action.id)
            repository.link_action_tool(recovery_action.id, recovery_action.tool)
            repository.link_action_run(recovery_action.id, fixture.run.id)
            repository.link_resolution_observed_change(
                fixture.resolution.id,
                recovery_action.id,
            )
        for pattern in patterns:
            repository.save_recovery_pattern(pattern)
        repository.ensure_recovery_pattern_vector_index()
        result = RecoveryPatternRetrievalService(
            repository,
            RecoveryPatternEmbedder(encoder=ConstantQueryEncoder()),
        ).retrieve(current_task, current_action, current_environment)

        assert result.selected_pattern is not None
        assert result.selected_pattern.id == patterns[0].id
        # Neo4j returns its native cosine-index score unchanged. For the
        # [0.80, 0.60] unit vector, this server reports (1 + cosine) / 2.
        assert result.selected_vector_score == pytest.approx(0.90, rel=1e-6)
        assert [candidate.pattern_id for candidate in result.eligible_candidates] == [
            patterns[0].id
        ]
        assert next(
            candidate for candidate in result.candidates if candidate.pattern_id == patterns[1].id
        ).rejection_reasons == ("not_strictly_historical",)
        assert next(
            candidate for candidate in result.candidates if candidate.pattern_id == patterns[2].id
        ).rejection_reasons == ("runtime_mismatch",)
    finally:
        for fixture in (earlier, future, incompatible):
            cleanup_fixture(repository, fixture.prefix)
