from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import Mock, patch

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
from graph_swarm.graph import queries
from graph_swarm.graph.neo4j_repository import EntityNotFoundError, Neo4jRepository

STARTED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
COMPLETED_AT = STARTED_AT + timedelta(seconds=1)


class FakeResult:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records


def make_repository() -> Neo4jRepository:
    with patch("graph_swarm.graph.neo4j_repository.GraphDatabase.driver", return_value=Mock()):
        return Neo4jRepository("uri", "username", "password", "database")


def make_action_pair() -> tuple[PlannedAction, ActionResult]:
    return (
        PlannedAction(
            id="action-001",
            run_id="run-001",
            task_id="task-001",
            tool="run_tests",
            operation="pytest",
            arguments={"paths": ["tests"], "options": {"quiet": True}},
            planned_at=STARTED_AT,
        ),
        ActionResult(
            action_id="action-001",
            tool_name="run_tests",
            success=True,
            exit_code=0,
            output="passed",
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
        ),
    )


def test_entity_writes_use_canonical_merge_identity_and_parameters() -> None:
    repository = make_repository()
    action, result = make_action_pair()
    entities = (
        (
            repository.save_run,
            Run(id="run-001", task_id="task-001", started_at=STARTED_AT),
            queries.SAVE_RUN,
            "run-001",
        ),
        (
            repository.save_task,
            Task(
                id="task-001",
                problem_statement="Fix the example repository so its tests pass.",
                family_id="family-001",
                repository="example/repository",
                chronological_index=1,
            ),
            queries.SAVE_TASK,
            "task-001",
        ),
        (repository.save_tool, Tool(name="run_tests"), queries.SAVE_TOOL, "run_tests"),
        (
            repository.save_environment,
            EnvironmentContext(
                id="environment-001",
                repository="example/repository",
                runtime="python-3.13",
            ),
            queries.SAVE_ENVIRONMENT,
            "environment-001",
        ),
        (
            repository.save_failure,
            FailureEpisode(
                id="failure-001",
                action_id="action-001",
                failure_type=FailureType.TEST_FAILURE,
                signature="signature",
                symptom="symptom",
                observed_at=STARTED_AT,
            ),
            queries.SAVE_FAILURE,
            "failure-001",
        ),
        (
            repository.save_resolution,
            Resolution(
                id="resolution-001",
                failure_id="failure-001",
                description="fix",
                status=ResolutionStatus.CANDIDATE,
            ),
            queries.SAVE_RESOLUTION,
            "resolution-001",
        ),
        (
            repository.save_outcome,
            Outcome(
                id="outcome-001",
                action_id="action-001",
                success=True,
                observed_at=COMPLETED_AT,
            ),
            queries.SAVE_OUTCOME,
            "outcome-001",
        ),
    )

    for save_method, entity, expected_query, identity in entities:
        with patch.object(repository, "execute_query") as execute_query:
            cast(Callable[[object], None], save_method)(entity)

        query = execute_query.call_args.args[0]
        parameters = execute_query.call_args.kwargs
        assert query == expected_query
        assert identity not in query
        assert identity in parameters.values()
        assert "MERGE" in query

    task_write = entities[1][1]
    with patch.object(repository, "execute_query") as execute_query:
        repository.save_task(task_write)
    assert execute_query.call_args.kwargs["problem_statement"] == (
        "Fix the example repository so its tests pass."
    )

    with patch.object(repository, "execute_query") as execute_query:
        repository.save_action(action, result)

    query = execute_query.call_args.args[0]
    parameters = execute_query.call_args.kwargs
    assert query == queries.SAVE_ACTION
    assert action.id not in query
    assert parameters["id"] == action.id
    assert parameters["arguments_json"] == '{"options":{"quiet":true},"paths":["tests"]}'


def test_save_action_rejects_mismatched_ids_before_database_write() -> None:
    repository = make_repository()
    action, result = make_action_pair()
    mismatched_result = result.model_copy(update={"action_id": "different-action"})

    with patch.object(repository, "execute_query") as execute_query:
        with pytest.raises(ValueError, match="must match"):
            repository.save_action(action, mismatched_result)

    execute_query.assert_not_called()


def test_relationship_writes_match_endpoints_and_merge_relationships() -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([{"linked": 1}]),
    ) as execute_query:
        repository.link_action_tool("action-001", "run_tests")

    query = execute_query.call_args.args[0]
    assert query == queries.LINK_ACTION_TOOL
    assert "MATCH" in query
    assert "MERGE (action)-[:USED]->(tool)" in query
    assert execute_query.call_args.kwargs == {
        "action_id": "action-001",
        "tool_name": "run_tests",
    }


def test_resolution_observed_change_relationship_is_idempotent_and_typed() -> None:
    repository = make_repository()

    with patch.object(
        repository,
        "execute_query",
        return_value=FakeResult([{"linked": 1}]),
    ) as execute_query:
        repository.link_resolution_observed_change("resolution-001", "action-001")

    assert execute_query.call_args.args[0] == queries.LINK_RESOLUTION_OBSERVED_CHANGE
    assert "MATCH" in execute_query.call_args.args[0]
    assert "MERGE (resolution)-[:OBSERVED_CHANGE]->(action)" in (
        execute_query.call_args.args[0]
    )
    assert execute_query.call_args.kwargs == {
        "resolution_id": "resolution-001",
        "action_id": "action-001",
    }


def test_relationship_write_rejects_missing_endpoint_and_empty_ids() -> None:
    repository = make_repository()

    with patch.object(repository, "execute_query", return_value=FakeResult([])):
        with pytest.raises(EntityNotFoundError, match="endpoint"):
            repository.link_action_run("action-001", "missing-run")

    with patch.object(repository, "execute_query") as execute_query:
        with pytest.raises(ValueError, match="action_id"):
            repository.link_action_run("", "run-001")

    execute_query.assert_not_called()

    with patch.object(repository, "execute_query", return_value=FakeResult([])):
        with pytest.raises(EntityNotFoundError, match="endpoint"):
            repository.link_resolution_observed_change(
                "resolution-001",
                "missing-action",
            )


def test_missing_incident_raises_clear_not_found_error() -> None:
    repository = make_repository()

    with patch.object(repository, "execute_query", return_value=FakeResult([])):
        with pytest.raises(EntityNotFoundError, match="was not found"):
            repository.get_incident_lineage("unknown-failure")


def test_lineage_query_converts_nodes_without_cross_product_duplicates() -> None:
    repository = make_repository()
    action, result = make_action_pair()
    repository.execute_query = Mock(
        side_effect=[
            FakeResult(
                [
                    {
                        "failure": {
                            "id": "failure-001",
                            "action_id": action.id,
                            "failure_type": "test_failure",
                            "signature": "signature",
                            "symptom": "symptom",
                            "observed_at": STARTED_AT.isoformat(),
                        }
                    }
                ]
            ),
            FakeResult(
                [
                    {
                        "environment": {
                            "id": "environment-001",
                            "repository": "example/repository",
                            "runtime": "python-3.13",
                            "versions_json": "{}",
                            "markers_json": "{}",
                        }
                    }
                ]
            ),
            FakeResult(
                [
                    {
                        "action": {
                            "id": action.id,
                            "run_id": action.run_id,
                            "task_id": action.task_id,
                            "tool": action.tool,
                            "operation": action.operation,
                            "arguments_json": '{"paths":["tests"],"options":{"quiet":true}}',
                            "planned_at": STARTED_AT.isoformat(),
                            "result_tool_name": result.tool_name,
                            "success": result.success,
                            "exit_code": result.exit_code,
                            "output": result.output,
                            "error": result.error,
                            "started_at": STARTED_AT.isoformat(),
                            "completed_at": COMPLETED_AT.isoformat(),
                        },
                        "task": {
                            "id": "task-001",
                            "problem_statement": "Fix the example repository so its tests pass.",
                            "family_id": "family-001",
                            "repository": "example/repository",
                            "chronological_index": 1,
                        },
                        "run": {
                            "id": "run-001",
                            "task_id": "task-001",
                            "started_at": STARTED_AT.isoformat(),
                        },
                    },
                    {
                        "action": {
                            "id": action.id,
                            "run_id": action.run_id,
                            "task_id": action.task_id,
                            "tool": action.tool,
                            "operation": action.operation,
                            "arguments_json": '{"paths":["tests"],"options":{"quiet":true}}',
                            "planned_at": STARTED_AT.isoformat(),
                            "result_tool_name": result.tool_name,
                            "success": result.success,
                            "exit_code": result.exit_code,
                            "output": result.output,
                            "error": result.error,
                            "started_at": STARTED_AT.isoformat(),
                            "completed_at": COMPLETED_AT.isoformat(),
                        },
                        "task": {
                            "id": "task-001",
                            "problem_statement": "Fix the example repository so its tests pass.",
                            "family_id": "family-001",
                            "repository": "example/repository",
                            "chronological_index": 1,
                        },
                        "run": {
                            "id": "run-001",
                            "task_id": "task-001",
                            "started_at": STARTED_AT.isoformat(),
                        },
                    },
                ]
            ),
            FakeResult([{"tool": {"name": "run_tests"}}]),
            FakeResult(
                [
                    {
                        "resolution": {
                            "id": "resolution-001",
                            "failure_id": "failure-001",
                            "description": "fix",
                            "status": "candidate",
                            "successful_observations": 0,
                            "failed_observations": 0,
                        },
                        "outcomes": [
                            {
                                "id": "outcome-001",
                                "action_id": action.id,
                                "success": True,
                                "tests_passed": 1,
                                "tests_failed": None,
                                "exit_code": 0,
                                "observed_at": COMPLETED_AT.isoformat(),
                            }
                        ],
                        "observed_changes": [
                            {
                                "id": action.id,
                                "run_id": action.run_id,
                                "task_id": action.task_id,
                                "tool": action.tool,
                                "operation": action.operation,
                                "arguments_json": (
                                    '{"options":{"quiet":true},"paths":["tests"]}'
                                ),
                                "planned_at": STARTED_AT.isoformat(),
                                "result_tool_name": result.tool_name,
                                "success": result.success,
                                "exit_code": result.exit_code,
                                "output": result.output,
                                "error": result.error,
                                "started_at": STARTED_AT.isoformat(),
                                "completed_at": COMPLETED_AT.isoformat(),
                            }
                        ],
                    }
                ]
            ),
        ]
    )

    lineage = repository.get_incident_lineage("failure-001")

    assert lineage.failure.id == "failure-001"
    assert lineage.run.id == "run-001"
    assert lineage.task.id == "task-001"
    assert lineage.task.problem_statement == "Fix the example repository so its tests pass."
    assert len(lineage.actions) == 1
    assert len(lineage.tools) == 1
    assert lineage.environment.id == "environment-001"
    assert len(lineage.resolutions) == 1
    assert len(lineage.outcomes) == 1
    assert len(lineage.recovery_actions) == 1
    assert lineage.recovery_actions[0].planned_action.arguments == action.arguments
    assert lineage.actions[0].result.action_id == lineage.actions[0].planned_action.id
